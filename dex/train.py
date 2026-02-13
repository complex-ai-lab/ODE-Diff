import argparse
import os
import torch
import torch.optim as optim
import pandas as pd
from dataloader import DataLoader
from sim_config import DataConfig, SEIRMConfig, OptimConfig, EvalConfig
from hybrid import HybridDecoder, EncoderLSTM, HybridModel

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
            "HillCure": torch.tensor(seirm_config.HillCure).to(device), 
            "HillPatho": torch.tensor(seirm_config.HillPatho).to(device), 
            "ec50_patho": torch.tensor(seirm_config.ec50_patho).to(device), 
            "emax_patho": torch.tensor(seirm_config.emax_patho).to(device), 
            "k_dexa": torch.tensor(seirm_config.k_dexa).to(device), 
            "k_discure_immunereact": torch.tensor(seirm_config.k_discure_immunereact).to(device), 
            "k_discure_immunity": torch.tensor(seirm_config.k_discure_immunity).to(device), 
            "k_disprog": torch.tensor(seirm_config.k_disprog).to(device), 
            "k_immune_disease": torch.tensor(seirm_config.k_immune_disease).to(device), 
            "k_immune_feedback": torch.tensor(seirm_config.k_immune_feedback).to(device), 
            "k_immune_off": torch.tensor(seirm_config.k_immune_off).to(device), 
            "k_immunity": torch.tensor(seirm_config.k_immunity).to(device), 
            "kel": torch.tensor(seirm_config.kel).to(device)
        }, 
        device=device
    )
    model = HybridModel(encoder, decoder).to(device)

    return model

def train_and_evaluate(model, data_loader, optim_config, device):
    optimizer = optim.Adam(model.parameters(), lr=optim_config.lr)
    model.train()
    model = model.to(device)

    input_length = 3
    save_interval = 1
    model_dir = "/scratch1/home/zhicao/ODE/model1/checkpoint_point"
    
    device = next(model.parameters()).device

    for epoch in range(optim_config.niters):
        total_loss_x_epoch = 0
        total_loss_y_epoch = 0
        
        for batch_idx, (x_data, y_data, a_data) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            a_data = a_data.to(device)
            
            loss_x, loss_y = model.loss(x_data, y_data, a_data, input_length)
            total_loss = loss_x + loss_y
            
            if torch.isnan(total_loss):
                print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss_x_epoch += loss_x.item()
            total_loss_y_epoch += loss_y.item()
            
            print(f'Epoch {epoch}, Sample {batch_idx}, Loss X: {loss_x.item()}, Loss Y: {loss_y.item()}')
        
        num_samples = len(data_loader)
        avg_loss_x = total_loss_x_epoch / num_samples
        avg_loss_y = total_loss_y_epoch / num_samples
        
        print(f'Epoch {epoch} Average Loss X: {avg_loss_x}, Average Loss Y: {avg_loss_y}')
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"hybrid_checkpoint_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

def run(result_csv, data_config, seirm_config, optim_config, device, model_weights=None):
    data_loader = DataLoader(result_csv, device)
    
    model = initialize_model(data_config, seirm_config, device=device)
    
    if model_weights is not None and os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location=device))
        print(f"Loaded model weights from {model_weights}")
    else:
        print("No pretrained model weights loaded.")
        
    train_and_evaluate(model, data_loader, optim_config=optim_config, device=device)
    
    print("Training completed")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference and Training")
    parser.add_argument("--result_csv", default="/scratch1/home/zhicao/ODE/dataset2/dataset2_train.csv", type=str)
    parser.add_argument("--model_weights", default="/scratch1/home/zhicao/ODE/model1/checkpoint_point/train.pth", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="0", type=str)
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    optim_config = OptimConfig()
    eval_config = EvalConfig()
    
    run(
        result_csv=args.result_csv,
        data_config=data_config,
        seirm_config=seirm_config,
        optim_config=optim_config,
        device=device,
        model_weights=args.model_weights
    )
