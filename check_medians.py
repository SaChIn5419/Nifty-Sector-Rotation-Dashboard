import pandas as pd
import numpy as np

df = pd.read_csv('data/combined_sectoral_data.csv', index_col=0, parse_dates=True)
pct = df.pct_change()

print("--- Data Check ---")
for col in ['Nifty Bank', 'Nifty IT', 'Nifty 50']:
    col_rets = pct[col].dropna()
    print(f"\n{col}:")
    print(f"Total valid returns: {len(col_rets)}")
    print(f"Zeros: {(col_rets == 0).sum()}")
    print(f"Median: {col_rets.median()}")
    print(f"Mean: {col_rets.mean()}")
    
    # Check by weekday
    temp = col_rets.to_frame()
    temp['Weekday'] = temp.index.day_name()
    weekday_median = temp.groupby('Weekday')[col].median()
    print("Weekday Medians:")
    print(weekday_median)
