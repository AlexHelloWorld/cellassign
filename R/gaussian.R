construct_gaussian_formulas <- function(G, C, P, B, shrinkage, min_delta, dirichlet_concentration, random_seed, tf, tfd){
  
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
  k <- tf$Variable(1, shape = shape(NULL), dtype = tf$float64)
  b <- tf$Variable(1, shape = shape(NULL), dtype = tf$float64)
  
  # calculate log expression mean
  # base mean
  base_mean_ng <- tf$transpose(tf$einsum('np,gp->gn', X_, beta) +
                                 tf$log(s_))
  
  base_mean_list <- list()
  for(c in seq_len(C)) base_mean_list[[c]] <- base_mean_ng
  base_mean_ngc <- tf$stack(base_mean_list, 2)
  
  # gene/cell_type specific expression: delta * rho
  delta <- tf$exp(delta_log)
  additional_mean_ngc <- tf$multiply(delta, rho_)
  
  # add base mean to gene/cell_type specific expression to generate mean (for the NB distribution)
  mu_ngc <- tf$add(base_mean_ngc, additional_mean_ngc, name = "adding_base_mean_to_delta_rho")
  
  # convert log expression mean to expression mean
  mu_ngc <- tf$exp(mu_ngc)
  
  # compute the standard deviation sigma of the normal distribution
  sigma_ngc <- k * mu_ngc + b
  
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