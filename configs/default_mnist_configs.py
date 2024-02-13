import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128 #387
  training.numworkers = 3
  training.n_epochs = 20
  training.snapshot_freq = 10
  training.eval_freq = 10
  training.distributed = True

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.generated_batch = 36   # batch size for sampling process
    
  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.checkpoint = 25
  evaluate.batch_size = 500 #1023
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000


  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.image_size = 28
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 1 

  # model
  config.model = model = ml_collections.ConfigDict()
  model.beta_min = 0.0001
  model.beta_max = 0.020
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.multiplier = 2.5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
