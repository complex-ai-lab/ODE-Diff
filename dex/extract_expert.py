import torch
import pandas as pd
import argparse
import os
import numpy as np
import random
from dataloader import DataLoader


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_with_noise(factual_csv_path, counterfactual_csv_path, noise_level=0.01):
    # Load factual and counterfactual data
    factual_df = pd.read_csv(factual_csv_path)
    counterfactual_df = pd.read_csv(counterfactual_csv_path)
    
    # Get dimensions
    n_patients = len(factual_df['patient_id'].unique())
    n_timesteps = len(factual_df['time'].unique())
    
    # Initialize arrays
    factual_vars = np.zeros((n_timesteps, n_patients, 4))
    counterfactual_vars = np.zeros((n_timesteps, n_patients, 4))
    
    # Process data for each patient
    for p in range(n_patients):
        patient_data = factual_df[factual_df['patient_id'] == p + 1]
        factual_vars[:, p, 0] = patient_data['Disease'].values
        factual_vars[:, p, 1] = patient_data['ImmuneReact'].values
        factual_vars[:, p, 2] = patient_data['Immunity'].values
        factual_vars[:, p, 3] = patient_data['Dose2'].values
        
        patient_data = counterfactual_df[counterfactual_df['patient_id'] == p + 1]
        counterfactual_vars[:, p, 0] = patient_data['Disease'].values
        counterfactual_vars[:, p, 1] = patient_data['ImmuneReact'].values
        counterfactual_vars[:, p, 2] = patient_data['Immunity'].values
        counterfactual_vars[:, p, 3] = patient_data['Dose2'].values
    
    # Add noise to the data
    if noise_level > 0:
        factual_noise = np.random.normal(0, noise_level, factual_vars.shape)
        counterfactual_noise = np.random.normal(0, noise_level, counterfactual_vars.shape)
        factual_vars = factual_vars + factual_noise
        counterfactual_vars = counterfactual_vars + counterfactual_noise
    
    return torch.tensor(factual_vars), torch.tensor(counterfactual_vars)

def save_predictions(factual_vars, counterfactual_vars, save_dir):
    factual_vars = factual_vars[:, :50, :]
    counterfactual_vars = counterfactual_vars[:, :50, :]
    
    factual_dir = os.path.join(save_dir, "factual")
    counterfactual_dir = os.path.join(save_dir, "counterfactual")
    os.makedirs(factual_dir, exist_ok=True)
    os.makedirs(counterfactual_dir, exist_ok=True)

    variable_names = ['disease', 'immunereact', 'immunity', 'dose2']

    for var_idx, var_name in enumerate(variable_names):
        factual_data = factual_vars[:, :, var_idx].T
        counterfactual_data = counterfactual_vars[:, :, var_idx].T

        factual_save_path = os.path.join(factual_dir, f'{var_name}_prediction.npy')
        counterfactual_save_path = os.path.join(counterfactual_dir, f'{var_name}_prediction.npy')
        np.save(factual_save_path, factual_data)
        np.save(counterfactual_save_path, counterfactual_data)
        print(f"Saved {var_name} factual predictions to {factual_save_path}")
        print(f"Saved {var_name} counterfactual predictions to {counterfactual_save_path}")

def generate_expert_predictions(factual_csv, counterfactual_csv, device, save_dir, noise_level=0.01):
    factual_vars, counterfactual_vars = load_data_with_noise(factual_csv, counterfactual_csv, noise_level)

    factual_vars = factual_vars.cpu().numpy()
    counterfactual_vars = counterfactual_vars.cpu().numpy()

    save_predictions(factual_vars, counterfactual_vars, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Save Expert Variables")
    parser.add_argument("--factual_csv", default="/scratch1/home/zhicao/ODE/dataset2/dataset2_expert_factual.csv", type=str)
    parser.add_argument("--counterfactual_csv", default="/scratch1/home/zhicao/ODE/dataset2/dataset2_expert_counterfactual.csv", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="0", type=str)
    parser.add_argument("--save_dir", default="/scratch1/home/zhicao/ODE/model1/expert_predictions", type=str)
    parser.add_argument("--base_noise", type=float, default=0.01, help="Base noise level")
    
    args = parser.parse_args()
    
    set_seed(2)
    print(f"Using seed: 0")
    
    noise_level = args.base_noise
    print(f"Using noise level: {noise_level:.6f}")
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    generate_expert_predictions(
        factual_csv=args.factual_csv,
        counterfactual_csv=args.counterfactual_csv,
        device=device,
        save_dir=args.save_dir,
        noise_level=noise_level
    )