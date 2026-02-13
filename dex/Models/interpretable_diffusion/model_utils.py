import math
import scipy
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from scipy.interpolate import interp1d


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


# normalization functions

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=1, padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd,  kernel_size=3, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, 
                 n_head, 
                 attn_pdrop=0.4, 
                 resid_pdrop=0.4, 
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 

        att = F.softmax(att, dim=-1) 
        att = self.attn_drop(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        att = att.mean(dim=1, keepdim=False) 

        y = self.resid_drop(self.proj(y))
        return y, att


class PredictionModel(nn.Module):
    def __init__(self, input_dim, seq_len, n_head, n_embd, attn_pdrop=0.5, resid_pdrop=0.5):
        super().__init__()
        self.input_dim = input_dim
        print('dropout: ',attn_pdrop, resid_pdrop)
        self.feature_proj = nn.Linear(input_dim, 32)
        self.acti = nn.SiLU()
        self.feature_proj_2 = nn.Linear(32, n_embd)
        self.attention = FullAttention(n_embd=n_embd, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.output = nn.Linear(n_embd, 64)

    def forward(self, x):
        x = self.feature_proj(x)  
        x = self.acti(x)
        x = self.feature_proj_2(x)
        attn_out, _ = self.attention(x) 
        x = attn_out.mean(dim=1)  
        return self.output(x)  



class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, label_dim=15):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, n_embd*2),
            nn.SiLU(),
            nn.Linear(n_embd*2, n_embd*2),
            nn.SiLU(),
            nn.Linear(n_embd*2, n_embd),
        )


    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift #10*scale
        return x
    

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, label_dim=72):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)
        self.label_proj = nn.Linear(label_dim, n_embd)


    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            label_emb = self.label_proj(label_emb) 
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x



def directional_sign_loss(
    diff_cf: torch.Tensor,     
    gt_f: torch.Tensor,        
    expert_cf: torch.Tensor,   
    expert_f: torch.Tensor     
) -> torch.Tensor:

    delta_pred = diff_cf - gt_f           
    delta_expert = expert_cf - expert_f  

    direction_pred = torch.tanh(1 * delta_pred)       
    direction_expert = torch.tanh(1 * delta_expert)   


    direction_loss = F.mse_loss(direction_pred, direction_expert)

    return direction_loss

def second_order_direction_loss(
    diff_cf: torch.Tensor,     
    gt_f: torch.Tensor,        
    expert_cf: torch.Tensor,   
    expert_f: torch.Tensor     
) -> torch.Tensor:

    delta_pred = diff_cf - gt_f          
    delta_expert = expert_cf - expert_f   

    d_delta_pred_dt = delta_pred[:, 1:, :] - delta_pred[:, :-1, :]      
    d_delta_expert_dt = delta_expert[:, 1:, :] - delta_expert[:, :-1, :]

    dir_pred = torch.tanh(1 * d_delta_pred_dt)         
    dir_expert = torch.tanh(1 * d_delta_expert_dt)     

    loss = F.mse_loss(dir_pred, dir_expert)

    return loss






def align_expert_by_peak_shift_after_t_numpy(
    expert_f: np.ndarray,   
    gt_f: np.ndarray,       
    expert_cf: np.ndarray,  
    t_fixed: int = 3        
) -> tuple[np.ndarray, np.ndarray]:

    B, T, D = expert_f.shape
    aligned_expert_f = np.zeros_like(expert_f)
    aligned_expert_cf = np.zeros_like(expert_cf)

    for b in range(B):
        for d in range(D):
            f_expert = expert_f[b, :, d]
            f_gt = gt_f[b, :, d]
            cf_expert = expert_cf[b, :, d]

            idx_peak_gt = np.argmax(f_gt)
            idx_peak_exp = np.argmax(f_expert)

            if idx_peak_exp <= t_fixed or idx_peak_gt <= t_fixed:
                aligned_expert_f[b, :, d] = f_expert
                aligned_expert_cf[b, :, d] = cf_expert
                continue

            scale = (idx_peak_gt - t_fixed) / (idx_peak_exp - t_fixed) if (idx_peak_exp - t_fixed) != 0 else 1.0

            original_time = np.arange(T)
            scaled_time = np.concatenate([
                original_time[:t_fixed],
                np.clip(t_fixed + (original_time[t_fixed:] - t_fixed) / scale, t_fixed, T - 1)
            ])

            interp_f = interp1d(original_time, f_expert, kind='linear', fill_value="extrapolate")
            interp_cf = interp1d(original_time, cf_expert, kind='linear', fill_value="extrapolate")

            aligned_expert_f[b, :, d] = interp_f(scaled_time)
            aligned_expert_cf[b, :, d] = interp_cf(scaled_time)

    return aligned_expert_f, aligned_expert_cf






