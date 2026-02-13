import pandas as pd
import numpy as np

def generate_covariates(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    hospitalization = df['Hospitalized_recovered'] + df['Hospitalized_deceased']
    
    infectious_symptomatic = df['Infectious_mild'] + df['Infectious_severe']
    
    covariate_df = pd.DataFrame({
        'y0_index': df['y0_index'],
        'hospitalization': hospitalization,
        'infectious_symptomatic': infectious_symptomatic
    })
    
    covariate_df.to_csv(output_csv, index=False)
    print(f"Generated covariates saved to {output_csv}")

input_csv = "/home/zhicao/ODE/data/weekly_10_data_test.csv"
output_csv = "/home/zhicao/ODE/data/weekly_10_covariate_test.csv"

generate_covariates(input_csv, output_csv)
