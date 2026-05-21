import ml_collections


def get_morris_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed = 42
  config.T = 1.0
  config.interval = 100
  config.train_method = 'joint'
  config.t0 = 0
  config.problem_name = 'morris'
  config.num_itr = 500
  config.eval_itr = 20
  config.forward_net = 'toy'
  config.backward_net = 'toy'

  # sampling
  config.samp_bs = 1000
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


