import torch
import os
from hybrid_real import HybridDecoder, EncoderLSTM, HybridModel
from sim_config import DataConfig, SEIRMConfig
from dataloader1 import DataLoader

def initialize_model(data_config, seirm_config, device):
    encoder = EncoderLSTM(
        input_dim=3,
        hidden_dim=data_config.latent_dim,
        output_dim_zx=data_config.latent_dim,
        output_dim_zy=data_config.latent_dim,
        output_dim_ze=data_config.expert_dim,
        device=device
    )

    decoder = HybridDecoder(
        zx_dim=data_config.latent_dim,
        zy_dim=data_config.latent_dim,
        ze_dim=data_config.expert_dim,
        action_dim=data_config.action_dim,
        y_dim=data_config.obs_dim,
        x_dim=data_config.obs_dim,
        step_size=data_config.step_size,
        params=seirm_config._asdict(),
        learnable_params={
            "beta": torch.tensor(seirm_config.beta).to(device), 
            "alpha": torch.tensor(seirm_config.alpha).to(device), 
            "gamma": torch.tensor(seirm_config.gamma).to(device), 
            "mu": torch.tensor(seirm_config.mu).to(device)
        }, 
        population=None,
        device=device
    )
    return HybridModel(encoder, decoder).to(device)

class ModifiedEncoderLSTM(EncoderLSTM):
    def forward(self, x, a, y):
        input_concat = torch.cat([x.unsqueeze(-1), a.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        
        hidden = None
        for t in reversed(range(input_concat.size(0))):
            obs = input_concat[t:t + 1, :]
            out, hidden = self.lstm(obs, hidden)
            
        hidden_state = out
        
        ze_init = self.g_eta(hidden_state)
        zx_init = self.g_xi(hidden_state)
        zy_input = torch.cat([hidden_state, zx_init], dim=-1)
        zy_init = self.g_zeta(zy_input)
        
        return zx_init, zy_init, ze_init, hidden_state

def main():
    device = torch.device("cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    
    data_loader = DataLoader(
        data_csv="/scratch1/home/zhicao/ODE/dataset1/data_real.csv",
        covariate_csv="/scratch1/home/zhicao/ODE/dataset1/data_real_covariate.csv",
        device=device
    )
    
    model = initialize_model(data_config, seirm_config, device)
    
    checkpoint_path = "/scratch1/home/zhicao/ODE/model/checkpoint_real/1.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    original_encoder = model.encoder
    modified_encoder = ModifiedEncoderLSTM(
        input_dim=3,
        hidden_dim=data_config.latent_dim,
        output_dim_zx=data_config.latent_dim,
        output_dim_zy=data_config.latent_dim,
        output_dim_ze=data_config.expert_dim,
        device=device
    )
    modified_encoder.load_state_dict(original_encoder.state_dict())
    model.encoder = modified_encoder
    all_hidden_states = []
    input_length = 27
    
    with torch.no_grad():
        for batch_idx, (x_data, y_data, a_data, population) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            a_data = a_data.to(device)
            
            x_input = x_data[:input_length]
            y_input = y_data[:input_length]
            a_input = a_data[:input_length]
            
            _, _, _, hidden_state = model.encoder(x_input, a_input, y_input)
            all_hidden_states.append(hidden_state)
    
    all_hidden_states = torch.cat(all_hidden_states, dim=0)
    
    all_hidden_states = all_hidden_states.cpu()
    
    save_path = "/scratch1/home/zhicao/ODE/model/lstm_embeddings_real.pt"
    torch.save(all_hidden_states, save_path)
    print(f"Hidden states saved to {save_path}")
    print(f"Hidden states shape: {all_hidden_states.shape}")

if __name__ == "__main__":
    main()