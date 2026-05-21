import ml_collections


def get_checkerboard_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.T = 1
  config.interval = 100
  config.train_method = 'joint'
  config.t0 = 0
  config.problem_name = 'checkerboard'
  config.num_itr = 1000
  config.eval_itr = 50
  config.forward_net = 'toy'
  config.backward_net = 'toy'

  # sampling
  # config.init = 102
  # config.final = 102    #here is the  number of sample
  config.init = 390
  config.final = 390    #here is the  number of sample
  config.sigma_min = 0.01
  config.sigma_max = 20

  # optimization
#   config.optim = optim = ml_collections.ConfigDict()
  config.weight_decay = 0
  config.optimizer = 'AdamW'
  config.lr = 1e-3
  config.lr_gamma = 0.9

  model_configs=None
  return config, model_configs


