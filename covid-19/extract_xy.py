import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from dataloader import DataLoader
from sim_config import DataConfig, SEIRMConfig
from hybrid import HybridDecoder, EncoderLSTM, HybridModel

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def run_inference_with_noise(data_csv, covariate_csv, model_path, data_config, seirm_config, device, noise_level=0.01):
    model = initialize_model(data_config, seirm_config, device)
    data_loader = DataLoader(data_csv, covariate_csv, device)
    
    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print("No pretrained model weights loaded.")
    model.eval()
    
    results = []
    with torch.no_grad():
        for idx, (x_data, y_data, a_data, population) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            a_data = a_data.to(device)
            
            input_noise = torch.randn_like(x_data) * noise_level
            x_data_noisy = x_data + input_noise
            
            action_noise = torch.randn_like(a_data) * noise_level
            a_data_noisy = a_data + action_noise
            
            model.decoder.population = population
            x_pred, y_pred = model(x_data_noisy, a_data_noisy, y_data, 15)
            test_loss_x, test_loss_y = model.loss(x_data, y_data, a_data, 15)
            
            print(f'Sample {idx}:')
            print(f'Test Loss X: {test_loss_x.item()}, Test Loss Y: {test_loss_y.item()}')
            
            results.append((y_pred, y_data, x_pred, x_data, idx))
    
    return results

def save_predictions(results, save_dir):
    y_preds = []
    x_preds = []
    
    for y_pred, _, x_pred, _, _ in results:
        y_preds.append(np.squeeze(y_pred.cpu().numpy()))
        x_preds.append(np.squeeze(x_pred.cpu().numpy()))
    
    y_preds = np.stack(y_preds, axis=0)
    x_preds = np.stack(x_preds, axis=0)
    
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'y_pred.npy'), y_preds)
    np.save(os.path.join(save_dir, 'x_pred.npy'), x_preds)
    
    print(f"Predictions (x_pred, y_pred) shape: {x_preds.shape}, {y_preds.shape}")
    print(f"Predictions saved to {save_dir}")

if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--data_csv", default="/scratch1/home/zhicao/ODE/dataset1/weekly_10_data_train.csv", type=str)
    parser.add_argument("--covariate_csv", default="/scratch1/home/zhicao/ODE/dataset1/weekly_10_covariate_train.csv", type=str)
    parser.add_argument("--model_path", default="/scratch1/home/zhicao/ODE/model/checkpoint_point/train.pth", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="0", type=str)
    parser.add_argument("--save_dir", default="/scratch1/home/zhicao/ODE/model/predictions_xy", type=str)
    parser.add_argument("--base_noise", type=float, default=0.000000001, help="Base noise level")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    
    seed = 2
    print(f"Using seed: {seed}")
    set_seed(seed)
    
    noise_level = args.base_noise
    print(f"Using noise level: {noise_level:.6f}")
    
    results = run_inference_with_noise(
        data_csv=args.data_csv,
        covariate_csv=args.covariate_csv,
        model_path=args.model_path,
        data_config=data_config,
        seirm_config=seirm_config,
        device=device,
        noise_level=noise_level
    )
    
    save_predictions(results, args.save_dir)

    
    