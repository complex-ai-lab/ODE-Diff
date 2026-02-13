import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F
from SEIRM import SEIRM
import torch.nn.init as init

class HybridODE(nn.Module):
    def __init__(self, zx_dim, zy_dim, ze_dim, action_dim, params, learnable_params, population, device=None):
        super(HybridODE, self).__init__()
        self.device = device or torch.device('cpu')
        
        # Initialize linear layers
        self.linear_zx = nn.Sequential(
            nn.Linear(zx_dim + zy_dim + action_dim, zx_dim).to(self.device),
            nn.Tanh(),
            nn.Linear(zx_dim, zx_dim).to(self.device),
            nn.Tanh(),
        ).to(self.device)
        
        self.linear_zy = nn.Sequential(
            nn.Linear(zy_dim + zx_dim + ze_dim + action_dim, zy_dim).to(self.device),
            nn.Tanh(),
            nn.Linear(zy_dim, zy_dim).to(self.device),
            nn.Tanh(),
        ).to(self.device)
        
        self.seirm = SEIRM(params, learnable_params, device=self.device)
        self.seirm.num_agents = population

        # Initialize parameters
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.ze_dim = ze_dim
        self.action_dim = action_dim
        
    def forward(self, t, state):
        # Extract zx, zy, ze, a from state
        zx = state[:, :self.zx_dim]  # (batch_size, zx_dim)
        zy = state[:, self.zx_dim:self.zx_dim + self.zy_dim]  # (batch_size, zy_dim)
        ze = state[:, self.zx_dim + self.zy_dim:self.zx_dim + self.zy_dim + self.ze_dim]  # (batch_size, ze_dim)
        a = state[:, -self.action_dim:]  # (batch_size, action_dim)
        ze_input = ze.squeeze()
        # Calculate the derivative of Ze(t)
        dze_dt = self.seirm(t, ze_input).unsqueeze(0)

        # Calculate the derivative of Zx(t) and apply tanh activation
        # time = torch.tensor([t]).unsqueeze(0)
        dzx_dt = self.linear_zx(torch.cat([zx, zy, a], dim=-1))
        dzx_dt = torch.clamp(dzx_dt, -1e6, 1e6)  # Prevent numerical overflow or underflow

        # Use updated Zx and Ze to calculate the derivative of Zy(t) and apply tanh activation
        dzy_dt = self.linear_zy(torch.cat([zy, zx + dzx_dt, ze + dze_dt, a], dim=-1))
        dzy_dt = torch.clamp(dzy_dt, -1e6, 1e6)

        # Return the combined derivatives
        dstate_dt = torch.cat([dzx_dt, dzy_dt, dze_dt, a], dim=-1)
        return dstate_dt


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_zx, output_dim_zy, output_dim_ze, device=None):
        super(EncoderLSTM, self).__init__()
        self.device = device or torch.device('cpu')
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        # Linear layers
        self.g_eta = nn.Sequential(
            nn.Linear(hidden_dim, output_dim_ze),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.g_xi = nn.Linear(hidden_dim, output_dim_zx).to(self.device)
        self.g_zeta = nn.Linear(hidden_dim + output_dim_zx, output_dim_zy).to(self.device)
        
        init.normal_(self.g_eta[0].weight, mean=0, std=0.01)
        init.normal_(self.g_xi.weight, mean=0, std=0.01)
        init.normal_(self.g_zeta.weight, mean=0, std=0.01)

    def forward(self, x, a, y):
        # Concatenate inputs
        input_concat = torch.cat([x.unsqueeze(-1), a.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)  # Shape: [n, 3]
        
        hidden = None
        for t in reversed(range(input_concat.size(0))):  # reversed range from n-1 to 0
            obs = input_concat[t:t + 1, :]  # Take one time step at a time, shape: [1, 3]
            out, hidden = self.lstm(obs, hidden)
            
        # Extract the hidden state from the last time step, which will be used for g_eta, g_xi, g_zeta
        # [1, hidden_size]
        hidden_state = out 
        
       # Ze(0) = softmax(gη(hidden_state))
        ze_init = self.g_eta(hidden_state)
        ze_init = ze_init * 10  # 直接将softmax的输出乘以10

        # Zx(0) = gξ(hidden_state)
        zx_init = self.g_xi(hidden_state)

        # Zy(0) = gζ([hidden_state, Zx(0)])
        zy_input = torch.cat([hidden_state, zx_init], dim=-1)
        zy_init = self.g_zeta(zy_input)

        return zx_init, zy_init, ze_init


class HybridDecoder(nn.Module):
    def __init__(self, zx_dim, zy_dim, ze_dim, action_dim, y_dim, x_dim, params, learnable_params, population,step_size, device=None):
        super(HybridDecoder, self).__init__()
        self.device = device or torch.device('cpu')
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.ze_dim = ze_dim
        self.step_size = step_size
        self.population = population

        self.hybrid_ode = HybridODE(zx_dim, zy_dim, ze_dim, action_dim, params, learnable_params, population, device=self.device)

        # Neural networks to generate x(t) and y(t)
        self.gy = nn.Sequential(
            nn.Linear(ze_dim + zy_dim + zx_dim + action_dim, y_dim),
            nn.Softplus()
        ).to(self.device)
        
        action_weights = self.gy[0].weight[:, -(action_dim):]
        init.normal_(action_weights, mean=-1.0, std=0.1)
            
        self.gx = nn.Sequential(
            nn.Linear(zx_dim + action_dim, x_dim),
            nn.Softplus()
        ).to(self.device)

        init.normal_(self.gy[0].weight, mean=0, std=0.01)
        init.normal_(self.gx[0].weight, mean=0, std=0.01)

    def forward(self, zx_init, zy_init, ze_init, a):
        # a: torch.Size([49])
        # zx_init: torch.Size([1, 5])
        self.hybrid_ode.seirm.num_agents = self.population
        remaining_steps = len(a)
        num_steps = remaining_steps // self.step_size
        
        # Create time series
        t = torch.arange(0, self.step_size * num_steps, self.step_size, device=self.device, dtype=torch.float32)

        # Combine zx_init, zy_init, ze_init, and a as the initial state
        a_init = a[0].unsqueeze(0)
        if a_init.dim() == 1:
            a_init = a_init.unsqueeze(-1)

        initial_state = torch.cat([zx_init, zy_init, ze_init, a_init], dim=-1)

        # Use torchdiffeq.odeint to solve ODE
        # state_trajectory = torchdiffeq.odeint(self.hybrid_ode, initial_state, t, method='dopri5')
        state_trajectory = torchdiffeq.odeint(
            self.hybrid_ode,
            initial_state,
            t,
            method='rk4',
            options={
                'step_size': 1
            }
        )

        # Extract zx, zy, and ze from state
        zx_output = state_trajectory[:, 0, :self.zx_dim]
        zy_output = state_trajectory[:, 0, self.zx_dim:self.zx_dim + self.zy_dim]
        ze_output = state_trajectory[:, 0, self.zx_dim + self.zy_dim:self.zx_dim + self.zy_dim + self.ze_dim]

        # Generate x(t) and y(t)
        # zx_output: torch.Size([49, 5])
        # a[input_length:].unsqueeze(-1)] torch.Size([49, 1])
        y_t = self.gy(torch.cat([ze_output, zy_output, zx_output, a.unsqueeze(-1)], dim=-1))
        x_t = self.gx(torch.cat([zx_output, a.unsqueeze(-1)], dim=-1))

        return x_t, y_t
    
class HybridModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(HybridModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, a, y, input_length):
        # Select input time points based on input_length
        x_input = x[:input_length]
        y_input = y[:input_length]
        a_input = a[:input_length]

        # Encoder step, using initial few time points based on input_length to generate initial values
        zx_init, zy_init, ze_init = self.encoder(x_input, a_input, y_input)
        
        # Decoder step, predict the values for the remaining time points
        x_pred, y_pred = self.decoder(zx_init, zy_init, ze_init, a)
        
        return x_pred, y_pred

    def loss(self, x_data, y_data, a_data, input_length):
        # Compute loss based on input_length
        # x_data: torch.Size([50])
        # a: torch.Size([50])
        x_pred, y_pred = self.forward(x_data, a_data, y_data, input_length)
        
        x_pred = x_pred.squeeze()
        y_pred = y_pred.squeeze()
        
        # x_pred: torch.Size([50])
        # y_pred: torch.Size([50])
        # loss_x = nn.MSELoss()(x_pred, x_data)
        # loss_y = nn.MSELoss()(y_pred, y_data)
        loss_x = torch.sqrt(nn.MSELoss()(x_pred, x_data))
        loss_y = torch.sqrt(nn.MSELoss()(y_pred, y_data))
        # rmse_loss_x = torch.sqrt(loss_x)
        # rmse_loss_y = torch.sqrt(loss_y)

        return loss_x, loss_y