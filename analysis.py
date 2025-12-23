import pandas as pd
import numpy as np
import os

def load_data(filepath='data/combined_sectoral_data.csv', benchmark_col='Nifty 50'):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # 0. Data Cleansing: Remove extreme glitches (Ratio-to-Benchmark)
    # Comparison to Nifty 50 over a 1-year window is extremely robust.
    if benchmark_col in df.columns:
        # Interpolate temporarily to create a continuous ratio for outlier detection
        bench_temp = df[benchmark_col].interpolate(limit_direction='both')
        for col in df.columns:
            if col == benchmark_col: continue
            
            col_temp = df[col].interpolate(limit_direction='both')
            ratio = col_temp / bench_temp
            
            # Use a wide 252-day (1 year) window to ensure the "true" ratio isn't lost in local glitches
            median_ratio = ratio.rolling(window=252, center=True, min_periods=1).median()
            deviation = (ratio - median_ratio).abs() / median_ratio
            
            # Any price causing >30% deviation from its 1-year median ratio is a glitch
            outliers = deviation > 0.30
            if outliers.any():
                df.loc[outliers, col] = np.nan

    # Now bridge gaps for real
    df = df.interpolate(method='linear', limit_area='inside').ffill().bfill()
    return df

def calculate_relative_rotation(df, benchmark_col='Nifty 50', window_rs=14, window_mom=14):
    """
    RRG Analysis:
    RS Ratio (J-Ratio) = (Price Index / Benchmark Index) * 100
    RS Momentum (J-Momentum) = Rate of Change of RS Ratio
    """
    rs_ratio = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col != benchmark_col:
            rs_ratio[col] = (df[col] / df[benchmark_col]) * 100

    # Normalizing and smoothing RS Ratio
    rs_ratio_smoothed = rs_ratio.apply(lambda x: x.rolling(window=window_rs).mean())
    
    # RS Momentum: Rate of change of the smoothed RS Ratio
    rs_momentum = (rs_ratio_smoothed / rs_ratio_smoothed.shift(window_mom)) * 100
    
    # Second smoothing for Momentum
    rs_momentum_smoothed = rs_momentum.apply(lambda x: x.rolling(window=window_mom).mean())

    return rs_ratio_smoothed, rs_momentum_smoothed

def calculate_momentum_ranking(df, period_months=[1, 3, 6, 12]):
    results = {}
    
    for months in period_months:
        days = months * 21 # approx trading days
        if len(df) > days:
            returns = (df.iloc[-1] / df.iloc[-days-1] - 1) * 100
            results[f'{months}M_Return'] = returns
            
    rankings_df = pd.DataFrame(results)
    return rankings_df

def calculate_monthly_heatmap_data(df):
    """Calculate returns per month and year for each sector."""
    all_heatmaps = {}
    for col in df.columns:
        # Monthly returns
        monthly_df = df[col].resample('ME').last()
        monthly_returns = monthly_df.pct_change() * 100
        
        heatmap_df = monthly_returns.to_frame()
        heatmap_df['Year'] = heatmap_df.index.year
        heatmap_df['Month'] = heatmap_df.index.month
        
        pivot_table = heatmap_df.pivot(index='Year', columns='Month', values=col)
        all_heatmaps[col] = pivot_table
        
    return all_heatmaps

def calculate_weekday_data(df):
    """Calculate average daily returns per weekday for each sector."""
    daily_returns = df.pct_change() * 100
    daily_returns['Weekday'] = daily_returns.index.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekday_analysis = daily_returns.groupby('Weekday').mean().reindex(weekday_order)
    
    return weekday_analysis

def main():
    if not os.path.exists('data/combined_sectoral_data.csv'):
        print("Data file not found. Please run data_fetcher.py first.")
        return

    df = load_data()
    print("Data loaded: ", df.shape)

    # 1. RRG Analysis
    rs_ratio, rs_momentum = calculate_relative_rotation(df)
    
    # 2. Momentum Ranking
    rankings = calculate_momentum_ranking(df)
    
    # 3. Monthly Heatmaps
    monthly_heatmaps = calculate_monthly_heatmap_data(df)
    
    # 4. Weekday Seasonality
    weekday_analysis = calculate_weekday_data(df)
    
    # Create results directory
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
        
    # Save rotation data
    rs_ratio.to_csv('analysis_results/rs_ratio.csv')
    rs_momentum.to_csv('analysis_results/rs_momentum.csv')
    rankings.to_csv('analysis_results/momentum_rankings.csv')
    weekday_analysis.to_csv('analysis_results/weekday_analysis.csv')
    
    # Save monthly performance for each sector in a separate subfolder
    heatmap_dir = 'analysis_results/monthly_heatmaps'
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    for sector, heat_df in monthly_heatmaps.items():
        heat_df.to_csv(f"{heatmap_dir}/{sector.replace(' ', '_').lower()}.csv")
    
    print("Analysis complete. Results saved in 'analysis_results/'.")

if __name__ == "__main__":
    main()
