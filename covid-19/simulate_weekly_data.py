import pandas as pd

input_file = '/home/zhicao/ODE/dataset1/daily_10_val_train.csv'
output_file = '/home/zhicao/ODE/dataset1/weekly_10_val_train.csv'

days_per_week = 7
total_weeks = 52
df = pd.read_csv(input_file)

num_cities = len(df['y0_index'].unique())
days_per_city = 364
weekly_data = []

for city_idx in range(num_cities):
    start_index = city_idx * days_per_city
    end_index = start_index + days_per_city
    
    daily_city_data = df.iloc[start_index:end_index]
    has_beta_decay = not (daily_city_data['beta'] == daily_city_data['beta'].iloc[0]).all()
    
    population = daily_city_data['Population'].iloc[0]

    for week in range(total_weeks):
        week_start = week * days_per_week
        week_end = week_start + days_per_week
        
        daily_week_data = daily_city_data.iloc[week_start:week_end]
        
        beta_avg = daily_week_data['beta'].mean()
        
        delta = daily_week_data['delta'].iloc[0]
        alpha = daily_week_data['alpha'].iloc[0]
        
        last_day_data = daily_week_data.drop(columns=['beta', 'Time', 'y0_index', 'delta', 'alpha', 'Population']).iloc[-1]
        
        a_value = 1 if (has_beta_decay and week >= 15) else 0
        
        week_data = {
            'y0_index': city_idx,
            'time': week + city_idx * total_weeks,
            'beta': beta_avg,
            'delta': delta,
            'alpha': alpha,
            'a': a_value,
            'Population': population,
            'Susceptible': last_day_data['Susceptible'],
            'Exposed': last_day_data['Exposed'],
            'Infectious_asymptomatic': last_day_data['Infectious_asymptomatic'],
            'Infectious_pre-symptomatic': last_day_data['Infectious_pre-symptomatic'],
            'Infectious_mild': last_day_data['Infectious_mild'],
            'Infectious_severe': last_day_data['Infectious_severe'],
            'Hospitalized_recovered': last_day_data['Hospitalized_recovered'],
            'Hospitalized_deceased': last_day_data['Hospitalized_deceased'],
            'Recovered': last_day_data['Recovered'],
            'Deceased': last_day_data['Deceased'],
        }
        
        weekly_data.append(week_data)

weekly_df = pd.DataFrame(weekly_data)
weekly_df.to_csv(output_file, index=False)
print(f'Weekly data saved to {output_file}')
