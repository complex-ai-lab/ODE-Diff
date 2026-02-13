import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_city_data():
    df = pd.read_csv('sub-est2022.csv')
    
    cities_df = df[
        (df['SUMLEV'] == 162) & 
        (df['ESTIMATESBASE2020'] >= 150000) &
        (df['ESTIMATESBASE2020'] < 200000)
    ].sort_values('ESTIMATESBASE2020', ascending=False)
    
    strict_states = [
        'California', 'New York', 'Illinois', 'Washington', 'Oregon',
        'Massachusetts', 'New Jersey', 'Connecticut', 'Rhode Island',
        'Delaware', 'Hawaii', 'Maryland', 'Michigan', 'Nevada',
        'New Mexico', 'Vermont', 'Virginia'
    ]
    
    strict_cities = [
        'New York city', 'Los Angeles city', 'Chicago city', 'San Francisco city',
        'Seattle city', 'Boston city', 'Portland city', 'San Jose city',
        'Denver city', 'Minneapolis city', 'Philadelphia city', 'Sacramento city',
        'San Diego city', 'Oakland city', 'Long Beach city'
    ]
    
    def get_mask_policy(row):
        if row['NAME'] in strict_cities or row['STNAME'] in strict_states:
            return 'Strict'
        return 'Relaxed'
    
    cities_df['mask_policy'] = cities_df.apply(get_mask_policy, axis=1)
    
    def assign_parameters(row):
        if row['mask_policy'] == 'Strict':
            delta = 0.15
            alpha = 0.3
        else:
            delta = 0.1
            alpha = 0.5
        return pd.Series({'delta': delta, 'alpha': alpha})
    
    parameter_df = cities_df.apply(assign_parameters, axis=1)
    cities_df = pd.concat([cities_df, parameter_df], axis=1)
    
    output_df = cities_df[['NAME', 'STNAME', 'ESTIMATESBASE2020', 'mask_policy', 'delta', 'alpha']]
    output_df.to_csv('large_cities_with_policies.csv', index=False)
    
    print(f"Total cities analyzed: {len(cities_df)}")
    print("\nMask policy distribution:")
    print(cities_df['mask_policy'].value_counts())
    
    return cities_df


def generate_initial_conditions():
    cities_df = load_city_data()
    populations = cities_df['ESTIMATESBASE2020'].tolist()
    policies = cities_df['mask_policy'].tolist()
    deltas = cities_df['delta'].tolist()
    alphas = cities_df['alpha'].tolist()
    
    initial_conditions = []
    normalized_populations = []
    for pop in populations:
        exposed = int(pop * 0.0015)
        ia = int(pop * 0.001)
        ip = int(pop * 0.0007)
        im = int(pop * 0.0005)
        is_ = int(pop * 0.0002)
        hr = int(pop * 0.00001)
        hd = int(pop * 0.000005)
        r = int(pop * 0.000005)
        d = int(pop * 0.000001)
        s = pop - (exposed + ia + ip + im + is_ + hr + hd + r + d)
        
        y0 = [s, exposed, ia, ip, im, is_, hr, hd, r, d]
        initial_conditions.append(torch.tensor(y0, dtype=torch.float32, device=DEVICE))
        normalized_populations.append(pop)
    
    return initial_conditions, policies, deltas, alphas, normalized_populations


class ODEFunc(nn.Module):
    def __init__(self, population, min_pop, max_pop):
        super(ODEFunc, self).__init__()
        self.N = torch.tensor(float(population), device=DEVICE)
        
        self.population_factor = 0.8 + (population - min_pop) * (0.2) / (max_pop - min_pop)
        self.Ca = nn.Parameter(torch.tensor(0.425, device=DEVICE))
        self.Cp = torch.tensor(1.0, device=DEVICE)
        self.Cm = torch.tensor(1.0, device=DEVICE)
        self.Cs = torch.tensor(1.0, device=DEVICE)
        self.alpha = torch.tensor(0.4875, device=DEVICE)
        self.delta = nn.Parameter(torch.tensor(0.1375, device=DEVICE))
        self.mu = torch.tensor(0.928125, device=DEVICE)
        self.gamma = torch.tensor(1/3.5, device=DEVICE)
        self.lambdaa = torch.tensor(1/7, device=DEVICE)
        self.lambdap = torch.tensor(1/1.5, device=DEVICE)
        self.lambdam = torch.tensor(1/5.5, device=DEVICE)
        self.lambdas = torch.tensor(1/5.5, device=DEVICE)
        self.rhor = torch.tensor(1/15, device=DEVICE)
        self.rhod = torch.tensor(1/13.3, device=DEVICE)
        self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))
        self.t = torch.tensor(0.0, device=DEVICE)


    def set_beta(self, beta):
        self.beta = nn.Parameter(torch.tensor(beta, device=DEVICE))


    def forward(self, t, y):
        S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y
        dt = t - self.t
        self.t = t
        dSE = self.population_factor * S * (1 - torch.exp(-self.beta * (self.Ca * Ia + self.Cp * Ip + self.Cm * Im + self.Cs * Is) * dt / self.N))
        dEIa = E * self.alpha * (1 - torch.exp(-self.gamma * dt))
        dEIp = E * (1 - self.alpha) * (1 - torch.exp(-self.gamma * dt))
        dIaR = Ia * (1 - torch.exp(-self.lambdaa * dt))
        dIpIm = Ip * self.mu * (1 - torch.exp(-self.lambdap * dt))
        dIpIs = Ip * (1 - self.mu) * (1 - torch.exp(-self.lambdap * dt))
        dImR = Im * (1 - torch.exp(-self.lambdam * dt))
        dIsHr = Is * self.delta * (1 - torch.exp(-self.lambdas * dt))
        dIsHd = Is * (1 - self.delta) * (1 - torch.exp(-self.lambdas * dt))
        dHrR = Hr * (1 - torch.exp(-self.rhor * dt))
        dHdD = Hd * (1 - torch.exp(-self.rhod * dt))

        dS = -dSE
        dE = dSE - dEIa - dEIp
        dIa = dEIa - dIaR
        dIp = dEIp - dIpIs - dIpIm
        dIm = dIpIm - dImR
        dIs = dIpIs - dIsHr - dIsHd
        dHr = dIsHr - dHrR
        dHd = dIsHd - dHdD
        dR = dHrR
        dD = dHdD
      
        dy = torch.stack([dS, dE, dIa, dIp, dIm, dIs, dHr, dHd, dR, dD])
        return dy
    
    def reset_t(self):
        self.t = torch.tensor(0.0, device=DEVICE)


def generate_inference_csv():
    initial_conditions, policies, deltas, alphas, populations = generate_initial_conditions()
    
    min_pop = min(populations)
    max_pop = max(populations)
    
    time_points = 363
    t = torch.arange(0, time_points + 1, dtype=torch.float32, device=DEVICE)
    
    def get_beta_schedule(initial_beta=0.5, lambda_=0.005):
        beta_schedule = []
        current_beta = initial_beta
        
        for _ in range(105):
            beta_schedule.append(current_beta)
        
        for _ in range(time_points + 1 - 105):
            current_beta = current_beta * math.exp(-lambda_)
            beta_schedule.append(current_beta)
        
        return torch.tensor(beta_schedule, dtype=torch.float32, device=DEVICE)
    
    results = []
    
    for i, (y0, policy, delta, alpha) in enumerate(zip(initial_conditions, policies, deltas, alphas)):
        total_population = populations[i]
        
        model = ODEFunc(total_population, min_pop, max_pop).to(DEVICE)
        model.delta = nn.Parameter(torch.tensor(delta, device=DEVICE))
        model.alpha = torch.tensor(alpha, device=DEVICE)
        
        if policy == 'Relaxed':
            beta_values = get_beta_schedule(initial_beta=0.5, lambda_=0.005)
            
            results.append({
                'Time': t[0].item(),
                'y0_index': i,
                'Population': total_population,
                'beta': beta_values[0].item(),
                'delta': delta,
                'alpha': alpha,
                'Susceptible': y0[0].item() * 10 / total_population,
                'Exposed': y0[1].item() * 10 / total_population,
                'Infectious_asymptomatic': y0[2].item() * 10 / total_population,
                'Infectious_pre-symptomatic': y0[3].item() * 10 / total_population,
                'Infectious_mild': y0[4].item() * 10 / total_population,
                'Infectious_severe': y0[5].item() * 10 / total_population,
                'Hospitalized_recovered': y0[6].item() * 10 / total_population,
                'Hospitalized_deceased': y0[7].item() * 10 / total_population,
                'Recovered': y0[8].item() * 10 / total_population,
                'Deceased': y0[9].item() * 10 / total_population,
            })
            
            for time_step in range(len(beta_values)-1):
                t_half = t[time_step:time_step+2].to(DEVICE)
                model.set_beta(beta_values[time_step].item())
                with torch.no_grad():
                    Y_pred = odeint(model, y0, t_half, method='rk4')
                y0 = Y_pred[-1]
                
                results.append({
                    'Time': t_half[1].item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': beta_values[time_step].item(),
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[-1, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[-1, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[-1, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[-1, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[-1, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[-1, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[-1, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[-1, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[-1, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[-1, 9].item() * 10 / total_population,
                })
        
        else:
            model.set_beta(0.5)
            with torch.no_grad():
                Y_pred = odeint(model, y0, t, method='rk4')
            
            for j, t_val in enumerate(t):
                results.append({
                    'Time': t_val.item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': 0.5,
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[j, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[j, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[j, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[j, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[j, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[j, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[j, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[j, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[j, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[j, 9].item() * 10 / total_population,
                })

    df = pd.DataFrame(results)
    df.to_csv('/home/zhicao/ODE/dataset1/daily_10_val_test.csv', index=False)
    return df

df = generate_inference_csv()
df.head()