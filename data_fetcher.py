import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_sectoral_data():
    # Define tickers for Nifty 50 and sectoral indices
    # Using 'NSE' suffix for indices on Yahoo Finance often requires '^' prefix or '.NS' for stocks.
    # For Nifty indices, '^NSEI' is standard for Nifty 50. 
    # Sectoral indices usually follow '^CNX...' or '^NSE...'
    
    tickers = {
        'Nifty 50': '^NSEI',
        'Nifty Bank': '^NSEBANK',
        'Nifty IT': '^CNXIT',
        'Nifty FMCG': '^CNXFMCG',
        'Nifty Auto': '^CNXAUTO',
        'Nifty Pharma': '^CNXPHARMA',
        'Nifty Metal': '^CNXMETAL',
        'Nifty Realty': '^CNXREALTY',
        'Nifty Energy': '^CNXENERGY',
        'Nifty Infra': '^CNXINFRA',
        'Nifty PSU Bank': '^CNXPSUBANK',
        'Nifty MNC': '^CNXMNC',
        'Nifty Service': '^CNXSERVICE',
        'Nifty Media': '^CNXMEDIA',
        'Nifty Commodities': '^CNXCMDT'
    }

    # Timeframe: 20 years back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20*365)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    all_data = {}
    
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                # Handle multi-index columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    # Usually (Price, Ticker)
                    if 'Close' in df.columns.get_level_values(0):
                        df_close = df['Close']
                    elif 'Adj Close' in df.columns.get_level_values(0):
                        df_close = df['Adj Close']
                    else:
                        df_close = df.iloc[:, 0] # fallback to first column
                    
                    # If still multi-index, flatten it (should be single column now)
                    if isinstance(df_close, pd.DataFrame):
                        df_close = df_close.iloc[:, 0]
                else:
                    if 'Adj Close' in df.columns:
                        df_close = df['Adj Close']
                    else:
                        df_close = df['Close']

                # Save individual CSV
                filename = f"{data_dir}/{name.replace(' ', '_').lower()}.csv"
                df.to_csv(filename)
                
                all_data[name] = df_close
                print(f"  Successfully saved {name}.")
            else:
                print(f"  Warning: No data found for {name} ({ticker}).")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")

    # Combine Adjusted Close prices into a single DataFrame for easier analysis
    if all_data:
        combined_df = pd.DataFrame(all_data)
        combined_df.to_csv(f"{data_dir}/combined_sectoral_data.csv")
        print(f"\nCombined data saved to {data_dir}/combined_sectoral_data.csv")
        print(f"Combined shape: {combined_df.shape}")
    else:
        print("\nNo data was downloaded.")

if __name__ == "__main__":
    fetch_sectoral_data()
