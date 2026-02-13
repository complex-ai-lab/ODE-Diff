import torch
import pandas as pd

class DataLoader:
    def __init__(self, data_csv, device):
        self.data_df = pd.read_csv(data_csv)
        self.device = device
        self.weeks_per_patient = 15
        self.num_patient = len(self.data_df) // self.weeks_per_patient
        self.current_idx = 0
        
    def __len__(self):
        return self.num_patient
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.num_patient:
            raise StopIteration
            
        city_idx = self.current_idx
        start_idx = city_idx * self.weeks_per_patient
        end_idx = start_idx + self.weeks_per_patient
        
        y_data = torch.tensor(
            self.data_df['measurement_1'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        x_data = torch.tensor(
            self.data_df['covariate_1'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        a_data = torch.tensor(
            self.data_df['dose'].values[start_idx:end_idx], 
            dtype=torch.float32
        ).to(self.device)
        
        self.current_idx += 1
        
        return x_data, y_data, a_data