import torch
import pandas as pd

class DataLoader:
    def __init__(self, data_csv, covariate_csv, device):
        self.data_df = pd.read_csv(data_csv)
        self.covariate_df = pd.read_csv(covariate_csv)
        self.device = device
        self.weeks_per_city = 52
        self.num_cities = len(self.data_df) // self.weeks_per_city
        self.current_idx = 0
        
    def __len__(self):
        return self.num_cities
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_cities:
            raise StopIteration
            
        city_idx = self.current_idx
        start_idx = city_idx * self.weeks_per_city
        end_idx = start_idx + self.weeks_per_city
        
        population = torch.tensor(
            self.data_df['Population'].values[start_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        y_data = torch.tensor(
            self.data_df['Deceased'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        x_data = torch.tensor(
            self.covariate_df['hospitalization'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        a_data = torch.tensor(
            self.data_df['a'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        self.current_idx += 1
        
        return x_data, y_data, a_data, population