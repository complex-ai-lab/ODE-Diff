import argparse
import os
import torch
import torch.optim as optim
from dataloader import DataLoader
from sim_config import DataConfig, SEIRMConfig, OptimConfig, EvalConfig
from model.hybrid_real import HybridDecoder, EncoderLSTM, HybridModel

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
    model = HybridModel(encoder, decoder).to(device)

    return model

def train_model(model, train_loader, optim_config, device, resume_path=None):
    optimizer = optim.Adam(model.parameters(), lr=optim_config.lr)
    model.train()
    model = model.to(device)

    start_epoch = 0
        
    if resume_path is not None and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Loaded model weights from {resume_path}")
        model.decoder.hybrid_ode.seirm.beta.data = torch.tensor(seirm_config.beta).to(device)
        model.decoder.hybrid_ode.seirm.alpha.data = torch.tensor(seirm_config.alpha).to(device)
        model.decoder.hybrid_ode.seirm.gamma.data = torch.tensor(seirm_config.gamma).to(device)
        model.decoder.hybrid_ode.seirm.mu.data = torch.tensor(seirm_config.mu).to(device)
    else:
        print("No pretrained model weights loaded.")

    input_length = 15
    save_interval = 1
    model_dir = "/home/zhicao/ODE/model/checkpoint_point"
    # model.decoder.gy[0].weight[:, -1:].fill_(-0.5)
    for epoch in range(start_epoch, optim_config.niters):
        total_loss_x_epoch = 0
        total_loss_y_epoch = 0
        
        # Training loop
        for batch_idx, (x_data, y_data, a_data, population) in enumerate(train_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            a_data = a_data.to(device)
            
            model.decoder.population = population
            
            loss_x, loss_y = model.loss(x_data, y_data, a_data, input_length)
            total_loss = loss_x + loss_y
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss_x_epoch += loss_x.item()
            total_loss_y_epoch += loss_y.item()
            
            print(f'Epoch {epoch}, Batch {batch_idx}:')
            print(f'Loss X: {loss_x.item():.4f}, Loss Y: {loss_y.item():.4f}, Total Loss: {total_loss.item():.4f}')
            
        # Calculate average training loss
        num_samples = len(train_loader)
        avg_loss_x = total_loss_x_epoch / num_samples
        avg_loss_y = total_loss_y_epoch / num_samples
        
        print(f'Epoch {epoch}:')
        print(f'Train - Avg Loss X: {avg_loss_x:.4f}, Avg Loss Y: {avg_loss_y:.4f}')
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"hybrid_checkpoint_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

def run(result_csv, noisy_deceased_csv, data_config, seirm_config, optim_config, device, resume_path=None):
    train_loader = DataLoader(result_csv, noisy_deceased_csv, device)
    model = initialize_model(data_config, seirm_config, device=device)
    train_model(model, train_loader, optim_config=optim_config, device=device, resume_path=resume_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("--result_csv", default="/home/zhicao/ODE/dataset1/weekly_10_data_train.csv", type=str)
    parser.add_argument("--noisy_deceased_csv", default="/home/zhicao/ODE/dataset1/weekly_10_covariate_train.csv", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str)
    parser.add_argument("--resume_path", type=str, default="/home/zhicao/ODE/model/checkpoint_point/1.pth")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    print(device)
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    optim_config = OptimConfig()
    eval_config = EvalConfig()
    
    run(
        result_csv=args.result_csv,
        noisy_deceased_csv=args.noisy_deceased_csv,
        data_config=data_config,
        seirm_config=seirm_config,
        optim_config=optim_config,
        device=device,
        resume_path=args.resume_path
    )