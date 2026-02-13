import torch
import torch.nn as nn
import numpy as np

class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()
        self.params = params
        self.device = device
        self.t = 0

    def reset_t(self):
        self.t = 0

class Roche(ODE):
    def __init__(self, params, learnable_params, device):
        super().__init__(params, device)
        self.HillCure = nn.Parameter(torch.tensor(learnable_params['HillCure'], device=device))
        self.HillPatho = nn.Parameter(torch.tensor(learnable_params['HillPatho'], device=device))
        self.ec50_patho = nn.Parameter(torch.tensor(learnable_params['ec50_patho'], device=device))
        self.emax_patho = nn.Parameter(torch.tensor(learnable_params['emax_patho'], device=device))
        self.k_dexa = nn.Parameter(torch.tensor(learnable_params['k_dexa'], device=device))
        self.k_discure_immunereact = nn.Parameter(torch.tensor(learnable_params['k_discure_immunereact'], device=device))
        self.k_discure_immunity = nn.Parameter(torch.tensor(learnable_params['k_discure_immunity'], device=device))
        self.k_disprog = nn.Parameter(torch.tensor(learnable_params['k_disprog'], device=device))
        self.k_immune_disease = nn.Parameter(torch.tensor(learnable_params['k_immune_disease'], device=device))
        self.k_immune_feedback = nn.Parameter(torch.tensor(learnable_params['k_immune_feedback'], device=device))
        self.k_immune_off = nn.Parameter(torch.tensor(learnable_params['k_immune_off'], device=device))
        self.k_immunity = nn.Parameter(torch.tensor(learnable_params['k_immunity'], device=device))
        self.kel = nn.Parameter(torch.tensor(learnable_params['kel'], device=device))
        
        self.beta = torch.tensor(0.1, device=device)
        self.initial_scale = torch.tensor(0.01, device=device)

    def get_initial_state(self):
        expert_init = np.random.exponential(scale=self.initial_scale.item(), size=4)
        return torch.from_numpy(expert_init).float().to(self.device)

    def forward(self, t, state, a):
        Disease = state[0]      # z4
        ImmuneReact = state[1]  # z1
        Immunity = state[2]     # z5
        Dose2 = state[3]        # z2
        
        Dose = 0
        
        dDisease = (
            Disease * self.k_disprog 
            - Disease * (Immunity ** self.HillCure) * self.k_discure_immunity
            - Disease * ImmuneReact * self.k_discure_immunereact
        )
        
        dImmuneReact = (
            Disease * self.k_immune_disease
            - ImmuneReact * self.k_immune_off
            + Disease * ImmuneReact * self.k_immune_feedback
            + (ImmuneReact ** self.HillPatho * self.emax_patho) 
            / (self.ec50_patho ** self.HillPatho + ImmuneReact ** self.HillPatho)
            - Dose2 * ImmuneReact * self.k_dexa
        )
        
        dImmunity = ImmuneReact * self.k_immunity
        dDose2 = self.kel * Dose - self.kel * Dose2
        
        dstate = torch.stack([dDisease, dImmuneReact, dImmunity, dDose2], 0)
        dstate = dstate - self.beta * state
        
        state = state + dstate
        state = torch.clamp(state, min=0.0, max=10.0)
        
        self.t = t
        return state