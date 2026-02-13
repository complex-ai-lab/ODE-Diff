from typing import NamedTuple

class SEIRMConfig(NamedTuple):
    beta: float = 0.5
    alpha: float = 0.4875
    gamma: float = 1/3.5
    mu: float = 0.928125
    num_agents: int = 1938000

class DataConfig(NamedTuple):
    n_sample: int = 1
    obs_dim: int = 1
    latent_dim: int = 64
    expert_dim: int = 5
    action_dim: int = 1
    step_size: int = 1
    sparsity: float = 0.5
    output_sigma: float = 0.1

# Example data configurations
dim8_config = DataConfig(obs_dim=40, latent_dim=8, output_sigma=0.2, sparsity=0.625)
dim12_config = DataConfig(obs_dim=80, latent_dim=12, output_sigma=0.2, sparsity=0.75)

class OptimConfig(NamedTuple):
    lr: float = 3e-5
    ode_method: str = "dopri5"
    niters: int = 100
    batch_size: int= 52
    shuffle: bool = True

class EvalConfig(NamedTuple):
    t0: int = 5
