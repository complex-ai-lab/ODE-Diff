import pickle
import sys
sys.path.append('../')
import numpy as np
import torch

import dataset2.dataloader_expert as dataloader
import sim_config

data_config = sim_config.DataConfig(n_sample=150)
n_sample = data_config.n_sample
obs_dim = data_config.obs_dim
latent_dim = data_config.latent_dim
action_dim = data_config.action_dim
t_max = data_config.t_max
step_size = data_config.step_size

p_remove = data_config.p_remove

output_sigma = 0.01

sparsity = data_config.sparsity
output_sparsity = 0.5
dose_max = 10

roche_config = sim_config.RochConfig(kel=1)

seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")

dg = dataloader.DataGeneratorRoche(
    n_sample,
    obs_dim,
    t_max,
    step_size,
    roche_config,
    output_sigma,
    dose_max,
    latent_dim,
    sparsity,
    p_remove=p_remove,
    output_sparsity=output_sparsity,
    device=device,
)
dg.generate_data()
dg.split_sample()

dg.export_to_csv2("dataset2_expert_counterfactual.csv")
