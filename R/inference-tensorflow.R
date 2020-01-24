

#' @keywords internal
#' Taken from https://github.com/tensorflow/tensorflow/issues/9162
entry_stop_gradients <- function(target, mask) {
  mask_h <- tf$logical_not(mask)
  mask <- tf$cast(mask, dtype = target$dtype)
  mask_h <- tf$cast(mask_h, dtype = target$dtype)

  tf$add(tf$stop_gradient(tf$multiply(mask_h, target)), tf$multiply(mask, target))
}

#' Estimate base expression for each marker gene
#' 
#' Given the normalized expression matrix, estimate base expression for each gene 
#' by fitting normalized expression data to gaussian mixture models
#' 
#' This function takes the normalized expression matrix and return a list of base expression estimates
#' The input list should be the complete normalized expression matrix
#' @param Y normalized expression matrix with all genes
#' 
#' @import mixdist
#' 
#' @return a numeric list of base expression estimates
#'
estimate_base_expression <- function(Y){
  baseExpression <- vector()
  for(i in 1:ncol(Y)){
    his <- hist(Y[, i], breaks = 50, plot = FALSE)
    df <- data.frame(mid=his$mids, cou=his$counts)
    
    medianExpression <- median(Y[, i])
    guess_mean <- c(medianExpression - 1, medianExpression + 1)
    guess_sigma <- c(1, 1)
    guess_dist <- "norm"
    
    fitpro <- tryCatch(mix(as.mixdata(df), mixparam(mu=guess_mean, sigma=guess_sigma), dist=guess_dist),
                       warning = function(cond){
                         message(paste0("Warning when fit gaussian mixture model: ", i))
                         NULL
                       },
                       error = function(cond){
                         message(paste0("Error when fit gaussian mixture model: ", i))
                         NULL
                         }
                       )
    
    if(is.null(fitpro)){
      baseExpression <- c(baseExpression, 0)
    }else{
      baseExpression <- c(baseExpression, min(fitpro$parameters[, "mu"]))
    }
  }
  baseExpression
}

#' construct cellassign formulas for optimization (Raw expression matrix. Negative binomial distribution hypothesis)
#' 
#' @import tensorflow
#' 
#' @return A list of variables for EM optimization
#' 
#' @keywords  internal
construct_tf_formulas <- function(G, C, P, B, minY, maxY, shrinkage, min_delta, dirichlet_concentration, random_seed, tf, tfd){
  B <- as.integer(B)
  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")
  
  # Variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C),
                                             minval = -2,
                                             maxval = 2,
                                             seed = random_seed,
                                             dtype = tf$float64),
                           dtype = tf$float64,
                           constraint = function(x) {
                             tf$clip_by_value(x,
                                              tf$constant(log(min_delta),
                                                          dtype = tf$float64),
                                              tf$constant(Inf, dtype = tf$float64))
                           })
  
  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))
  
  beta <- tf$Variable(tf$random_normal(shape(G,P),
                                       mean = 0,
                                       stddev = 1,
                                       seed = random_seed,
                                       dtype = tf$float64),
                      dtype = tf$float64)
  
  theta_logit <- tf$Variable(tf$random_normal(shape(C),
                                              mean = 0,
                                              stddev = 1,
                                              seed = random_seed,
                                              dtype = tf$float64),
                             dtype = tf$float64)
  
  
  ## Spline variables
  a <- tf$exp(tf$Variable(tf$zeros(shape = B, dtype = tf$float64)))
  
  ## initiate b
  basis_means_fixed <- seq(from = minY, to = maxY, length.out = B)
  b_init <- 2 * (basis_means_fixed[2] - basis_means_fixed[1])^2
  b <- tf$exp(tf$constant(rep(-log(b_init), B), dtype = tf$float64))
  
  # calculate log expression mean
  # mean
  base_mean_ng <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) +
                                 tf$log(s_))
  
  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean_ng
  base_mean_ngc <- tf$stack(base_mean_list, 2)
  
  # gene/cell_type specific expression: delta * rho*
  delta <- tf$exp(delta_log)
  additional_mean_ngc <- tf$multiply(delta, rho_)
  
  # add base mean to gene/cell_type specific expression to generate mean (for the NB distribution)
  mu_ngc <- tf$add(base_mean_ngc, additional_mean_ngc, name = "adding_base_mean_to_delta_rho")
  
  # calculate NB dispersion
  basis_means <- tf$constant(basis_means_fixed, dtype = tf$float64)
  LOWER_BOUND <- 1e-10
  
  mu_cng <- tf$transpose(mu_ngc, shape(2,0,1))
  mu_cngb <- tf$tile(tf$expand_dims(mu_cng, axis = 3L), c(1L, 1L, 1L, B))
  phi_cng <- tf$reduce_sum(a * tf$exp(-b * tf$square(mu_cngb - basis_means)), 3L) + LOWER_BOUND
  phi_ngc <- tf$transpose(phi_cng, shape(1,2,0))
  
  # convert log expression mean to expression mean
  mu_ngc <- tf$exp(mu_ngc)
  
  # calculate NB parameters
  # probability of success
  nb_probability <- mu_ngc / (mu_ngc + phi_ngc)
  # NB PDF
  nb_pdf <- tfd$NegativeBinomial(probs = nb_probability, total_count = phi_ngc)
  
  # calculate negative binomial probabilities
  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y_ngc <- tf$stack(Y_tensor_list, axis = 2)
  
  y_log_prob_ncg <- tf$transpose(nb_pdf$log_prob(Y_ngc), shape(0, 2, 1))
  
  # sum y_log_prob over gene
  y_log_prob_nc <- tf$reduce_sum(y_log_prob_ncg, 2L)
  
  # theta: probability of cell n belonging to cell type c
  theta_log <- tf$nn$log_softmax(theta_logit)
  
  y_z_log_prob_nc <- y_log_prob_nc + theta_log
  y_z_log_prob_cn <- tf$transpose(y_z_log_prob_nc, shape(1, 0))
  
  y_z_log_prob_n <- tf$reshape(tf$reduce_logsumexp(y_z_log_prob_cn, 0L), shape(1,-1))
  
  # E-step gamma
  gamma <- tf$transpose(tf$exp(y_z_log_prob_cn - y_z_log_prob_n))
  
  # M-step formula: Q to be minimized
  gamma_fixed <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,C), name = "gamma_fixed")
  Q <- -tf$einsum('nc,cn->', gamma_fixed, y_z_log_prob_cn)
  
  ## add theta log probs and delta log probs to Q
  
  # delta priors
  if (shrinkage) {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
    
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
    Q <- Q + delta_log_prob
  }
  
  # theta priors
  THETA_LOWER_BOUND <- 1e-20
  theta_log_prior <- tfd$Dirichlet(concentration = tf$constant(dirichlet_concentration, dtype = tf$float64))
  theta_log_prob <- -theta_log_prior$log_prob(tf$exp(theta_log) + THETA_LOWER_BOUND)
  
  Q <- Q + delta_log_prob
  
  # marginal log likelihood fo monitoring EM convergence
  L_y = tf$reduce_sum(tf$reduce_logsumexp(y_z_log_prob_cn, 0L))
  
  L_y <- L_y - theta_log_prob
  if (shrinkage) {
    L_y <- L_y - delta_log_prob
    list(Q = Q, L_y = L_y, gamma = gamma, 
         delta = delta, beta = beta, phi = phi_ngc, mu_ngc = mu_ngc, a = a, theta = tf$exp(theta_log), 
         ld_mean = delta_log_mean, ld_var = delta_log_variance)
  }else{
    list(Q = Q, L_y = L_y, gamma = gamma, 
         delta = delta, beta = beta, phi = phi_ngc, mu_ngc = mu_ngc, a = a, theta = tf$exp(theta_log))
  }
}

#' construct cellassign formulas for optimization (batch-corrected & normalized expression matrix. Normal distribution hypothesis)
#' 
#' @import tensorflow
#' 
#' @return A list of variables for EM optimization
#' 
#' @keywords  internal
construct_gaussian_formulas <- function(G, C, P, B, base_mean_estimates, shrinkage, min_delta, dirichlet_concentration, random_seed, tf, tfd){
  
  B <- as.integer(B)
  # Data placeholders
  Y_ <- tf$placeholder(tf$float64, shape = shape(NULL, G), name = "Y_")
  X_ <- tf$placeholder(tf$float64, shape = shape(NULL, P), name = "X_")
  s_ <- tf$placeholder(tf$float64, shape = shape(NULL), name = "s_")
  rho_ <- tf$placeholder(tf$float64, shape = shape(G,C), name = "rho_")
  
  # Variables
  delta_log <- tf$Variable(tf$random_uniform(shape(G,C),
                                             minval = -2,
                                             maxval = 2,
                                             seed = random_seed,
                                             dtype = tf$float64),
                           dtype = tf$float64,
                           constraint = function(x) {
                             tf$clip_by_value(x,
                                              tf$constant(log(min_delta),
                                                          dtype = tf$float64),
                                              tf$constant(Inf, dtype = tf$float64))
                           })
  
  # Stop gradient for irrelevant entries of delta_log
  delta_log <- entry_stop_gradients(delta_log, tf$cast(rho_, tf$bool))
  
  beta <- tf$Variable(tf$random_normal(shape(G,P),
                                       mean = 0,
                                       stddev = 1,
                                       seed = random_seed,
                                       dtype = tf$float64),
                      dtype = tf$float64)
  
  theta_logit <- tf$Variable(tf$random_normal(shape(C),
                                              mean = 0,
                                              stddev = 1,
                                              seed = random_seed,
                                              dtype = tf$float64),
                             dtype = tf$float64)
  
  # normal distribution standard deviation 
  k <- tf$Variable(1.0, dtype = tf$float64, constraint = function(x) {
    tf$clip_by_value(x,
                     tf$constant(0, dtype = tf$float64),
                     tf$constant(Inf, dtype = tf$float64))
  })
  b <- tf$Variable(0.01, dtype = tf$float64, constraint = function(x) {
    tf$clip_by_value(x,
                     tf$constant(0.0001, dtype = tf$float64),
                     tf$constant(Inf, dtype = tf$float64))
  })
  
  # add gene expression base
  gene_base <- tf$constant(base_mean_estimates, dtype = tf$float64)
  
  # calculate log expression mean
  # base mean
  base_mean_ng <- gene_base + tf$transpose(tf$einsum('np,gp->gn', X_, beta) + tf$log(s_))
  
  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean_ng
  base_mean_ngc <- tf$stack(base_mean_list, 2)
  
  # gene/cell_type specific expression: delta * rho
  delta <- tf$exp(delta_log)
  additional_mean_ngc <- tf$multiply(delta, rho_)
  
  # add base mean to gene/cell_type specific expression to generate mean (for the NB distribution)
  mu_ngc <- tf$add(base_mean_ngc, additional_mean_ngc, name = "adding_base_mean_to_delta_rho")
  
  # compute the standard deviation sigma of the normal distribution
  sigma_ngc <- tf$sqrt(tf$nn$relu(k * mu_ngc) + b)
  
  # normal distribution
  gaussian_pdf <- tfd$Normal(loc = mu_ngc, scale = sigma_ngc)
  
  # calculate probability
  Y_tensor_list <- list()
  for(c in seq_len(C)) Y_tensor_list[[c]] <- Y_
  Y_ngc <- tf$stack(Y_tensor_list, axis = 2)
  
  y_log_prob_ncg <- tf$transpose(gaussian_pdf$log_prob(Y_ngc), shape(0, 2, 1))
  
  # sum y_log_prob over gene
  y_log_prob_nc <- tf$reduce_sum(y_log_prob_ncg, 2L)
  
  # theta: probability of cell n belonging to cell type c
  theta_log <- tf$nn$log_softmax(theta_logit)
  
  y_z_log_prob_nc <- y_log_prob_nc + theta_log
  y_z_log_prob_cn <- tf$transpose(y_z_log_prob_nc, shape(1, 0))
  
  y_z_log_prob_n <- tf$reshape(tf$reduce_logsumexp(y_z_log_prob_cn, 0L), shape(1,-1))
  
  # E-step gamma
  gamma <- tf$transpose(tf$exp(y_z_log_prob_cn - y_z_log_prob_n))
  
  # M-step formula: Q to be minimized
  gamma_fixed <- tf$placeholder(dtype = tf$float64, shape = shape(NULL,C), name = "gamma_fixed")
  Q <- -tf$einsum('nc,cn->', gamma_fixed, y_z_log_prob_cn)
  
  ## add theta log probs and delta log probs to Q
  
  # delta priors
  if (shrinkage) {
    delta_log_mean <- tf$Variable(0, dtype = tf$float64)
    delta_log_variance <- tf$Variable(1, dtype = tf$float64) # May need to bound this or put a prior over this
    
    delta_log_prior <- tfd$Normal(loc = delta_log_mean * rho_,
                                  scale = delta_log_variance)
    delta_log_prob <- -tf$reduce_sum(delta_log_prior$log_prob(delta_log))
    Q <- Q + delta_log_prob
  }
  
  # theta priors
  THETA_LOWER_BOUND <- 1e-20
  theta_log_prior <- tfd$Dirichlet(concentration = tf$constant(dirichlet_concentration, dtype = tf$float64))
  theta_log_prob <- -theta_log_prior$log_prob(tf$exp(theta_log) + THETA_LOWER_BOUND)
  
  Q <- Q + delta_log_prob
  
  # marginal log likelihood fo monitoring EM convergence
  L_y = tf$reduce_sum(tf$reduce_logsumexp(y_z_log_prob_cn, 0L))
  
  L_y <- L_y - theta_log_prob
  if (shrinkage) {
    L_y <- L_y - delta_log_prob
    list(Q = Q, L_y = L_y, gamma = gamma, 
         delta = delta, beta = beta, sigma_ngc = sigma_ngc, mu_ngc = mu_ngc, theta = tf$exp(theta_log), 
         ld_mean = delta_log_mean, ld_var = delta_log_variance)
  }else{
    list(Q = Q, L_y = L_y, gamma = gamma, 
         delta = delta, beta = beta, sigma_ngc = sigma_ngc, mu_ngc = mu_ngc, theta = tf$exp(theta_log))
  }
  
}


#' cellassign inference in tensorflow, semi-supervised version
#'
#' @import tensorflow
#'
#' @return A list of MLE cell type calls, MLE parameter estimates,
#' and log likelihoods during optimization.
#'
#' @keywords internal
inference_tensorflow <- function(Y,
                                 rho,
                                 s,
                                 X,
                                 G,
                                 C,
                                 N,
                                 P,
                                 B = 10,
                                 shrinkage,
                                 distribution = "Negative Binomial",
                                 verbose = FALSE,
                                 n_batches = 1,
                                 rel_tol_adam = 1e-4,
                                 rel_tol_em = 1e-4,
                                 max_iter_adam = 1e5,
                                 max_iter_em = 20,
                                 learning_rate = 1e-4,
                                 random_seed = NULL,
                                 min_delta = 2,
                                 dirichlet_concentration = rep(1e-2, C),
                                 threads = 0) {
  tf <- tf$compat$v1
  tf$disable_v2_behavior()
  
  tfp <- reticulate::import('tensorflow_probability')
  tfd <- tfp$distributions
  
  tf$reset_default_graph()
  
  if(distribution == "Negative Binomial"){
    tf_formulas <- construct_tf_formulas(G, C, P, B, min(Y), max(Y), shrinkage, min_delta, dirichlet_concentration, random_seed, tf, tfd)
  }else if(distribution == "Normal"){
    base_mean_estimates <- estimate_base_expression(Y)
    tf_formulas <- construct_gaussian_formulas(G, C, P, B, base_mean_estimates, shrinkage, min_delta, dirichlet_concentration, random_seed, tf, tfd)
  }else{
    stop("Only support expression distribution: 'Negative Binomial' and 'Normal'")
  }
  
  
  gamma <- tf_formulas$gamma
  Q <- tf_formulas$Q
  L_y <- tf_formulas$L_y
  
  optimizer <- tf$train$AdamOptimizer(learning_rate=learning_rate)
  train <- optimizer$minimize(Q)
  
  # Split the data
  splits <- split(sample(seq_len(N), size = N, replace = FALSE), seq_len(n_batches))
  
  # Start the graph and inference
  session_conf <- tf$ConfigProto(intra_op_parallelism_threads = as.integer(threads),
                                 inter_op_parallelism_threads = as.integer(threads))
  sess <- tf$Session(config = session_conf)
  init <- tf$global_variables_initializer()
  sess$run(init)
  
  
  fd_full <- dict("Y_:0" = Y, "X_:0" = X, "s_:0" = s, "rho_:0" = rho)
  
  log_liks <- ll_old <- sess$run(L_y, feed_dict = fd_full)
  
  for(i in seq_len(max_iter_em)) {
    ll <- 0 # log likelihood for this "epoch"
    for(b in seq_len(n_batches)) {
      
      fd <- dict("Y_:0" = Y[splits[[b]], ],
                 "X_:0" = X[splits[[b]], , drop = FALSE],
                 "s_:0" = s[splits[[b]]],
                 "rho_:0" = rho)
      
      g <- sess$run(gamma, feed_dict = fd)
      
      # M-step
      gfd <- dict("Y_:0" = Y[splits[[b]], ],
                  "X_:0" = X[splits[[b]], , drop = FALSE],
                  "s_:0" = s[splits[[b]]],
                  "rho_:0" = rho,
                  "gamma_fixed:0" = g)
      
      Q_old <- sess$run(Q, feed_dict = gfd)
      Q_diff <- rel_tol_adam + 1
      mi = 0
      
      while(mi < max_iter_adam && Q_diff > rel_tol_adam) {
        mi <- mi + 1
        
        sess$run(train, feed_dict = gfd)
        
        if(mi %% 20 == 0) {
          if (verbose) {
            message(paste(mi, sess$run(Q, feed_dict = gfd)))
          }
          Q_new <- sess$run(Q, feed_dict = gfd)
          if(is.nan(Q_new)){
            print(sess$run(tf_formulas$mu_ngc, feed_dict = gfd))
            print(sess$run(tf_formulas$sigma_ngc, feed_dict = gfd))
          }
          Q_diff = -(Q_new - Q_old) / abs(Q_old)
          Q_old <- Q_new
          
        }
      } # End gradient descent
      
      l_new = sess$run(L_y, feed_dict = gfd) # Log likelihood for this "epoch"
      ll <- ll + l_new
    }
    
    ll_diff <- (ll - ll_old) / abs(ll_old)
    
    if(verbose) {
      message(sprintf("%i\tL old: %f; L new: %f; Difference (%%): %f",
                      mi, ll_old, ll, ll_diff))
    }
    ll_old <- ll
    log_liks <- c(log_liks, ll)
    
    if (ll_diff < rel_tol_em) {
      break
    }
  }
  
  # Finished EM - peel off final values
  if(distribution == "Negative Binomial"){
    variable_list <- list(tf_formulas$delta, tf_formulas$beta, tf_formulas$phi, tf_formulas$gamma, tf_formulas$mu_ngc, tf_formulas$theta)
    variable_names <- c("delta", "beta", "phi", "gamma", "mu_ngc", "theta")
  }else{
    variable_list <- list(tf_formulas$delta, tf_formulas$beta, tf_formulas$sigma_ngc, tf_formulas$gamma, tf_formulas$mu_ngc, tf_formulas$theta)
    variable_names <- c("delta", "beta", "sigma_ngc", "gamma", "mu_ngc", "theta")
  }
  
  mle_params <- sess$run(variable_list, feed_dict = fd_full)
  names(mle_params) <- variable_names
  sess$close()
  
  mle_params$delta[rho == 0] <- 0
  
  if(is.null(colnames(rho))) {
    colnames(rho) <- paste0("cell_type_", seq_len(ncol(rho)))
  }
  colnames(mle_params$gamma) <- colnames(rho)
  rownames(mle_params$delta) <- rownames(rho)
  colnames(mle_params$delta) <- colnames(rho)
  rownames(mle_params$beta) <- rownames(rho)
  names(mle_params$theta) <- colnames(rho)
  
  
  cell_type <- get_mle_cell_type(mle_params$gamma)
  
  rlist <- list(
    cell_type = cell_type,
    mle_params = mle_params,
    lls=log_liks
  )
  
  rlist
  
}
