import torch
import torch.nn as nn

class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()
        self.params = params
        self.device = device
        self.num_agents = 1938000
        self.t = 0

    def reset_t(self):
        self.t = 0


class SEIRM(ODE):
    def __init__(self, params, learnable_params, device):
        super().__init__(params,device)
        self.beta = nn.Parameter(torch.tensor(learnable_params['beta'], device=device))
        self.alpha = nn.Parameter(torch.tensor(learnable_params['alpha'], device=device))
        self.gamma = nn.Parameter(torch.tensor(learnable_params['gamma'], device=device))
        self.mu = nn.Parameter(torch.tensor(learnable_params['mu'], device=device))

    def forward(self, t, state):
        """
        Computes ODE states via equations       
        state is the array of state value (S,E,I,R,M)
        Returns: state per 10 people
        """
        state = state * self.num_agents/10
        
        dSE = self.beta * state[0] * state[2] / self.num_agents
        dEI = self.alpha * state[1] 
        dIR = self.gamma * state[2] 
        dIM = self.mu * state[2] 

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        
        dR  = dIR
        dM  = dIM

        dstate = torch.stack([dS, dE, dI, dR, dM], 0)
        state = state + dstate
        
        dstate_per_10 = (dstate / self.num_agents) * 10
        
        self.t = t
        return dstate_per_10
