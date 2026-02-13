import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import random

def set_seed(seed=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_city_data():
    df = pd.read_csv('/scratch1/home/zhicao/ODE/dataset1/sub-est2022.csv')
    cities_df = df[(df['SUMLEV'] == 162) & (df['ESTIMATESBASE2020'] >= 200000)].sort_values('ESTIMATESBASE2020', ascending=False)
    
    strict_states = ['California', 'New York', 'Illinois', 'Washington', 'Oregon', 'Massachusetts', 'New Jersey', 'Connecticut', 'Rhode Island',
                    'Delaware', 'Hawaii', 'Maryland', 'Michigan', 'Nevada', 'New Mexico', 'Vermont', 'Virginia']
    strict_cities = ['New York city', 'Los Angeles city', 'Chicago city', 'San Francisco city', 'Seattle city', 'Boston city', 'Portland city',
                    'San Jose city', 'Denver city', 'Minneapolis city', 'Philadelphia city', 'Sacramento city', 'San Diego city', 'Oakland city', 'Long Beach city']
    
    cities_df['mask_policy'] = cities_df.apply(lambda row: 'Strict' if row['NAME'] in strict_cities or row['STNAME'] in strict_states else 'Relaxed', axis=1)
    cities_df['delta'] = cities_df['mask_policy'].apply(lambda x: 0.15 if x == 'Strict' else 0.1)
    cities_df['alpha'] = cities_df['mask_policy'].apply(lambda x: 0.3 if x == 'Strict' else 0.5)
    
    return cities_df

class SEIRM(nn.Module):
    def __init__(self, population, min_pop, max_pop):
        super(SEIRM, self).__init__()
        self.N = torch.tensor(float(population), device=DEVICE)
        self.population_factor = 0.8 + (population - min_pop) * (0.2) / (max_pop - min_pop)
        self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))
        self.alpha = torch.tensor(0.4875, device=DEVICE)
        self.gamma = torch.tensor(1/3.5, device=DEVICE)
        self.mu = torch.tensor(0.928125, device=DEVICE)
        self.t = torch.tensor(0.0, device=DEVICE)

    def forward(self, t, y):
        S, E, I, R, M = y
        dt = t - self.t
        self.t = t
        
        dSE = self.population_factor * S * (1 - torch.exp(-self.beta * I * dt / self.N))
        dEI = E * self.alpha * (1 - torch.exp(-self.gamma * dt))
        dIR = I * (1 - self.mu) * (1 - torch.exp(-1/7 * dt))
        dIM = I * self.mu * (1 - torch.exp(-1/7 * dt))

        return torch.stack([-dSE, dSE - dEI, dEI - dIR - dIM, dIR, dIM])

    def reset_t(self):
        self.t = torch.tensor(0.0, device=DEVICE)

def generate_predictions(population, min_pop, max_pop, noise_level=0.01):
    exposed = int(population * 0.0015)
    infectious = int(population * 0.001)
    recovered = int(population * 0.000005)
    mortality = int(population * 0.000001)
    susceptible = population - (exposed + infectious + recovered + mortality)
    
    y0 = torch.tensor([susceptible, exposed, infectious, recovered, mortality], 
                     dtype=torch.float32, device=DEVICE)
    t = torch.arange(0, 364, dtype=torch.float32, device=DEVICE)
    
    if noise_level > 0:
        noise = torch.randn_like(y0) * noise_level * y0
        y0 = y0 + noise
        y0 = torch.clamp(y0, min=0)
    
    model = SEIRM(population, min_pop, max_pop).to(DEVICE)
    with torch.no_grad():
        Y_pred_no_treatment = odeint(model, y0, t, method='rk4')
    
    model_treatment = SEIRM(population, min_pop, max_pop).to(DEVICE)
    Y_pred_treatment = []
    current_y = y0
    current_beta = 0.5
    lambda_ = 0.005
    
    for day in range(364):
        if day >= 105:
            current_beta = current_beta * math.exp(-lambda_)
            model_treatment.beta = nn.Parameter(torch.tensor(current_beta, device=DEVICE))
        
        t_current = torch.tensor([float(day), float(day + 1)], device=DEVICE)
        with torch.no_grad():
            y_current = odeint(model_treatment, current_y, t_current, method='rk4')
            Y_pred_treatment.append(y_current[-1])
            current_y = y_current[-1]
    
    Y_pred_treatment = torch.stack(Y_pred_treatment)
    return Y_pred_no_treatment, Y_pred_treatment

def get_train_test_treatment(city_idx, train_data, test_data, num_weeks):
    city_test_data = test_data[city_idx * num_weeks:(city_idx + 1) * num_weeks]
    city_train_data = train_data[city_idx * num_weeks:(city_idx + 1) * num_weeks]
    train_treatment = (city_train_data['a'] == 1).any()
    test_treatment = (city_test_data['a'] == 1).any()
    return train_treatment, test_treatment

def main(noise_level=0.01):
    set_seed(2)
    
    cities_df = load_city_data()
    populations = cities_df['ESTIMATESBASE2020'].tolist()
    min_pop = min(populations)
    max_pop = max(populations)
    
    num_cities = len(populations)
    num_weeks = 52
    variable_names = ['susceptible', 'exposed', 'infectious', 'recovered', 'mortality']

    test_data = pd.read_csv('/scratch1/home/zhicao/ODE/dataset1/weekly_10_data_test.csv')
    train_data = pd.read_csv('/scratch1/home/zhicao/ODE/dataset1/weekly_10_data_train.csv')

    all_weekly_train = [np.zeros((num_cities, num_weeks)) for _ in range(5)]
    all_weekly_test = [np.zeros((num_cities, num_weeks)) for _ in range(5)]
    
    for i, population in enumerate(populations):
        Y_pred_no_treatment, Y_pred_treatment = generate_predictions(population, min_pop, max_pop, noise_level)
        
        weekly_no_treatment = np.zeros((num_weeks, 5))
        weekly_treatment = np.zeros((num_weeks, 5))
        for week in range(num_weeks):
            end_idx = (week + 1) * 7 - 1
            weekly_no_treatment[week] = Y_pred_no_treatment[end_idx].cpu().numpy()
            weekly_treatment[week] = Y_pred_treatment[end_idx].cpu().numpy()
        
        weekly_no_treatment = weekly_no_treatment / population
        weekly_treatment = weekly_treatment / population
        
        weekly_growth_no_treatment = np.zeros_like(weekly_no_treatment)
        weekly_growth_treatment = np.zeros_like(weekly_treatment)
        weekly_growth_no_treatment[0] = weekly_no_treatment[0]
        weekly_growth_treatment[0] = weekly_treatment[0]
        for week in range(1, num_weeks):
            weekly_growth_no_treatment[week] = weekly_no_treatment[week] - weekly_no_treatment[week-1]
            weekly_growth_treatment[week] = weekly_treatment[week] - weekly_treatment[week-1]
        
        train_treatment, test_treatment = get_train_test_treatment(i, train_data, test_data, num_weeks)
        
        for var_idx in range(5):
            if train_treatment:
                all_weekly_train[var_idx][i, :] = weekly_growth_treatment[:, var_idx]
            else:
                all_weekly_train[var_idx][i, :] = weekly_growth_no_treatment[:, var_idx]
            if test_treatment:
                all_weekly_test[var_idx][i, :] = weekly_growth_treatment[:, var_idx]
            else:
                all_weekly_test[var_idx][i, :] = weekly_growth_no_treatment[:, var_idx]
        
        print(f"Completed processing city {i}")

    save_dir_train = '/scratch1/home/zhicao/ODE/model/predictions_expert/train'
    save_dir_test = '/scratch1/home/zhicao/ODE/model/predictions_expert/test'
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)
    
    for var_idx, var_name in enumerate(variable_names):
        save_path_train = os.path.join(save_dir_train, f'{var_name}_weekly.npy')
        save_path_test = os.path.join(save_dir_test, f'{var_name}_weekly.npy')
        np.save(save_path_train, all_weekly_train[var_idx])
        np.save(save_path_test, all_weekly_test[var_idx])
        print(f"Saved {var_name}_weekly.npy to train and test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Generate Expert Predictions")
    parser.add_argument("--base_noise", type=float, default=0.01, help="Base noise level")
    
    args = parser.parse_args()
    
    print(f"Using noise level: {args.base_noise:.6f}")
    main(noise_level=args.base_noise)
