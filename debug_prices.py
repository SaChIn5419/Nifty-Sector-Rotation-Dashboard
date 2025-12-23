import pandas as pd
df = pd.read_csv('data/combined_sectoral_data.csv', index_col=0)
print(f"Total rows: {len(df)}")
for col in ['Nifty Bank', 'Nifty IT', 'Nifty 50']:
    print(f"\n--- {col} ---")
    print(f"Unique prices: {df[col].nunique()}")
    print(f"First 10 values:\n{df[col].head(10)}")
    valid_data = df[col].dropna()
    print(f"Valid data points: {len(valid_data)}")
    print(f"Value counts of first 100 valid prices:\n{valid_data.head(100).value_counts().head(5)}")
