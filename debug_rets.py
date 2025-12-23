import pandas as pd
import numpy as np

df = pd.read_csv('data/combined_sectoral_data.csv', index_col=0)
for col in ['Nifty Bank', 'Nifty IT', 'Nifty 50']:
    rets = df[col].pct_change()
    print(f"\n--- {col} ---")
    print(f"Total days: {len(rets)}")
    print(f"Zero returns: {(rets == 0).sum()}")
    print(f"NaN returns: {rets.isna().sum()}")
    print(f"Unique non-zero returns count: {len(rets[rets != 0].dropna().unique())}")
    print("Top 5 returns:")
    print(rets.value_counts().head(5))
