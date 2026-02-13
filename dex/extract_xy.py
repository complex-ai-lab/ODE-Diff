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
    return HybridModel(encoder, decoder).to(device)

def plot_predictions(x_true, y_true, x_pred, y_pred, case_idx, save_dir):
    """Plot and save predictions for both x and y variables"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(save_dir, 'x_pred'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'y_pred'), exist_ok=True)
    
    # Plot X predictions
    plt.figure(figsize=(10, 6))
    plt.plot(x_true.cpu().numpy(), 'k-', label='Ground Truth', linewidth=2)
    plt.plot(x_pred.cpu().numpy(), 'r--', label='Prediction', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Disease Level')
    plt.title(f'Disease Progression - Case {case_idx}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 3.5)
    plt.savefig(os.path.join(save_dir, 'x_pred', f'case_{case_idx}.png'))
    plt.close()
    
    # Plot Y predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.cpu().numpy(), 'k-', label='Ground Truth', linewidth=2)
    plt.plot(y_pred.cpu().numpy(), 'r--', label='Prediction', linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Immune Response')
    plt.title(f'Immune Response - Case {case_idx}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 3.5)
    plt.savefig(os.path.join(save_dir, 'y_pred', f'case_{case_idx}.png'))
    plt.close()
    

def plot_all_cities(all_x_true, all_y_true, all_x_pred, all_y_pred, save_dir):
    """Plot all cities together for both x and y variables"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    for i in range(len(all_x_true)):
        plt.plot(all_x_true[i].cpu().numpy(), 'k-', alpha=0.3)
        plt.plot(all_x_pred[i].cpu().numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('Disease Level')
    plt.title('Disease Progression - All Cases')
    plt.grid(True)
    plt.plot([], [], 'k-', label='Ground Truth')
    plt.plot([], [], 'r--', label='Prediction')
    plt.legend()
    plt.ylim(-1, 3.5)
    plt.savefig(os.path.join(save_dir, 'all_cities_x_pred.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    for i in range(len(all_y_true)):
        plt.plot(all_y_true[i].cpu().numpy(), 'k-', alpha=0.3)
        plt.plot(all_y_pred[i].cpu().numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('Immune Response')
    plt.title('Immune Response - All Cases')
    plt.grid(True)
    plt.plot([], [], 'k-', label='Ground Truth')
    plt.plot([], [], 'r--', label='Prediction')
    plt.legend()
    plt.ylim(-1, 3.5)
    plt.savefig(os.path.join(save_dir, 'all_cities_y_pred.png'))
    plt.close()

def run_inference_with_noise(model, data_loader, device, save_dir, input_length=3, noise_level=0.01):
    """Run inference with added noise for robustness"""
    model.eval()
    
    y_preds = []
    x_preds = []
    
    with torch.no_grad():
        for idx, (x_data, y_data, a_data) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            a_data = a_data.to(device)
            
            # Add small random noise to input data
            input_noise = torch.randn_like(x_data) * noise_level
            x_data_noisy = x_data + input_noise
            
            # Add small random noise to action data
            action_noise = torch.randn_like(a_data) * noise_level
            a_data_noisy = a_data + action_noise
            
            x_pred, y_pred = model(x_data_noisy, a_data_noisy, y_data, input_length)
            
            y_preds.append(y_pred.cpu().numpy())
            x_preds.append(x_pred.cpu().numpy())
            
            print(f'Sample {idx} processed')
    
    y_preds = np.stack(y_preds, axis=0)
    x_preds = np.stack(x_preds, axis=0)

    y_preds = y_preds[:50]
    x_preds = x_preds[:50]
    
    y_preds = y_preds[:, :15]
    x_preds = x_preds[:, :15]
    
    y_preds = np.squeeze(y_preds, axis=-1)
    x_preds = np.squeeze(x_preds, axis=-1)
    
    save_predictions(y_preds, x_preds, save_dir)

def save_predictions(y_pred, x_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(save_dir, 'x_pred.npy'), x_pred)
    
    print(f"Prediction results saved to {save_dir}")
    print(f"Data shapes:")
    print(f"y_pred: {y_pred.shape}")
    print(f"x_pred: {x_pred.shape}")

def main():
    parser = argparse.ArgumentParser("Model Inference")
    parser.add_argument("--test_csv", default="/scratch1/home/zhicao/ODE/dataset2/dataset2_train.csv", type=str)
    parser.add_argument("--model_path", default="/scratch1/home/zhicao/ODE/model1/checkpoint_point/train1.pth", type=str)
    parser.add_argument("--save_dir", default="/scratch1/home/zhicao/ODE/model1/predictions_xy_train", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="0", type=str)
    parser.add_argument("--base_noise", type=float, default=0.01, help="Base noise level")
    
    args = parser.parse_args()
    
    set_seed(2)
    print(f"Using seed: 0")
    
    noise_level = args.base_noise
    print(f"Using noise level: {noise_level:.6f}")
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    
    model = initialize_model(data_config, seirm_config, device)
    
    if args.model_path is not None and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights: {args.model_path}")
    else:
        print("No pretrained model weights loaded")
    
    data_loader = DataLoader(args.test_csv, device)
    run_inference_with_noise(model, data_loader, device, args.save_dir, noise_level=noise_level)

if __name__ == "__main__":
    main()