import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nifty Sectoral Rotation Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/combined_sectoral_data.csv', index_col=0, parse_dates=True)
    
    # --- Robust Data Cleansing (Ratio-to-Benchmark) ---
    # This removes glitches by comparing sectors to Nifty 50 over a 1-year rolling window
    benchmark_col = 'Nifty 50'
    if benchmark_col in df.columns:
        bench_temp = df[benchmark_col].interpolate(limit_direction='both')
        for col in df.columns:
            if col == benchmark_col: continue
            col_temp = df[col].interpolate(limit_direction='both')
            ratio = col_temp / bench_temp
            # 252-day window catches glitches even if they are surrounded by NaNs
            median_ratio = ratio.rolling(window=252, center=True, min_periods=1).median()
            outliers = ((ratio - median_ratio).abs() / median_ratio) > 0.30
            df.loc[outliers, col] = np.nan
    
    # Bridge middle gaps and handle outer NAs
    df = df.interpolate(method='linear', limit_area='inside').ffill().bfill()
    return df

@st.cache_data
def load_rrg_data():
    rs_ratio = pd.read_csv('analysis_results/rs_ratio.csv', index_col=0, parse_dates=True)
    rs_momentum = pd.read_csv('analysis_results/rs_momentum.csv', index_col=0, parse_dates=True)
    # Smooth out any gaps in RRG trails
    rs_ratio = rs_ratio.interpolate(method='linear').ffill().bfill()
    rs_momentum = rs_momentum.interpolate(method='linear').ffill().bfill()
    return rs_ratio, rs_momentum

df_price = load_data()
rs_ratio, rs_momentum = load_rrg_data()

# --- CALCULATE METRICS ---
def calculate_metrics(prices, benchmark_col='Nifty 50'):
    returns = prices.pct_change().dropna(how='all')
    results = {}
    
    bench_returns = returns[benchmark_col].dropna()
    
    for col in prices.columns:
        col_prices = prices[col].dropna()
        if len(col_prices) < 2:
            continue
            
        # 1. KPI & Return Stats
        total_return = (col_prices.iloc[-1] / col_prices.iloc[0]) - 1
        num_years = (col_prices.index[-1] - col_prices.index[0]).days / 365.25
        cagr = (1 + total_return)**(1/num_years) - 1 if num_years > 0 else 0
        
        col_returns = col_prices.pct_change().dropna()
        vol = col_returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.06) / vol if vol > 0 else 0 
        
        # Max Drawdown
        cum_returns = (1 + col_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # 2. CAPM (Alpha/Beta) relative to Nifty 50
        # Align returns for regression
        aligned = pd.concat([bench_returns, col_returns], axis=1).dropna()
        if len(aligned) > 5:
            slope, intercept, _, _, _ = stats.linregress(aligned.iloc[:,0], aligned.iloc[:,1])
            beta = slope
            alpha_annualized = (intercept * 252)
        else:
            beta, alpha_annualized = np.nan, np.nan
        
        results[col] = {
            "CAGR": f"{cagr:.2%}",
            "Volatility": f"{vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Beta": f"{beta:.2f}",
            "Alpha (Ann)": f"{alpha_annualized:.2%}",
            "Skewness": f"{col_returns.skew():.2f}",
            "Kurtosis": f"{col_returns.kurtosis():.2f}",
            "Last Price": f"{col_prices.iloc[-1]:,.2f}"
        }
    return pd.DataFrame(results).T

# ... (sidebar code same)

# --- SIDEBAR FILTERS ---
st.sidebar.title("📊 Filter Analysis")
date_range = st.sidebar.date_input("Select Date Range", 
                                   value=[df_price.index.min(), df_price.index.max()],
                                   min_value=df_price.index.min(),
                                   max_value=df_price.index.max())

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = df_price.index.min(), df_price.index.max()

# Filter data
filtered_df = df_price.loc[start_date:end_date].dropna(axis=1, how='all')
# Correctly rebase
for col in filtered_df.columns:
    first_valid = filtered_df[col].first_valid_index()
    if first_valid:
        filtered_df[col] = (filtered_df[col] / filtered_df.loc[first_valid, col]) * 100

# --- MAIN DASHBOARD ---
st.title("🇮🇳 Indian Sectoral Rotation Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔄 RRG Rotation", "📊 Metrics & Stats", "🔥 Seasonality"])

with tab1:
    st.subheader("Cumulative Performance (Rebased to 100)")
    selected_sectors = st.multiselect("Select Sectors", options=filtered_df.columns.tolist(), default=['Nifty 50', 'Nifty Bank', 'Nifty IT'])
    
    fig_perf = px.line(filtered_df[selected_sectors], 
                       labels={"value": "Performance Index", "index": "Date"},
                       log_y=True)
    fig_perf.update_layout(height=600)
    st.plotly_chart(fig_perf, use_container_width=True)

with tab2:
    st.subheader("Relative Rotation Graph (RRG)")
    rrg_date = st.select_slider("Time Travel - Snapshot Date", 
                                options=rs_ratio.index.strftime('%Y-%m-%d').tolist(),
                                value=rs_ratio.index[-1].strftime('%Y-%m-%d'))
    
    trail_length = st.slider("Trail Length (Weeks)", 1, 52, 12)
    
    # Get subset for RRG
    snap_idx = rs_ratio.index.get_loc(rrg_date)
    start_idx = max(0, snap_idx - trail_length)
    
    ratio_subset = rs_ratio.iloc[start_idx:snap_idx+1]
    momen_subset = rs_momentum.iloc[start_idx:snap_idx+1]
    
    fig_rrg = go.Figure()
    
    # Add Quadrants
    x_range = [min(ratio_subset.min().min(), 98)-1, max(ratio_subset.max().max(), 102)+1]
    y_range = [min(momen_subset.min().min(), 98)-1, max(momen_subset.max().max(), 102)+1]
    
    # Better RRG Layout
    fig_rrg.add_vline(x=100, line_width=1.5, line_color="black", opacity=0.8)
    fig_rrg.add_hline(y=100, line_width=1.5, line_color="black", opacity=0.8)
    
    # Add Colorful Quadrant Shapes (Background)
    fig_rrg.add_shape(type="rect", x0=100, x1=x_range[1], y0=100, y1=y_range[1], fillcolor="green", opacity=0.08, layer="below", line_width=0)
    fig_rrg.add_shape(type="rect", x0=x_range[0], x1=100, y0=100, y1=y_range[1], fillcolor="blue", opacity=0.08, layer="below", line_width=0)
    fig_rrg.add_shape(type="rect", x0=x_range[0], x1=100, y0=y_range[0], y1=100, fillcolor="red", opacity=0.08, layer="below", line_width=0)
    fig_rrg.add_shape(type="rect", x0=100, x1=x_range[1], y0=y_range[0], y1=100, fillcolor="orange", opacity=0.08, layer="below", line_width=0)

    for col in ratio_subset.columns:
        fig_rrg.add_trace(go.Scatter(x=ratio_subset[col], y=momen_subset[col],
                                     mode='lines+markers',
                                     name=col,
                                     marker=dict(size=[4]*len(ratio_subset[:-1]) + [15],
                                                 line=dict(width=1, color='DarkSlateGrey')),
                                     hovertemplate="<b>%{text}</b><br>RS Ratio: %{x:.2f}<br>RS Momentum: %{y:.2f}",
                                     text=[col]*len(ratio_subset)))

    fig_rrg.update_layout(
        xaxis=dict(title="RS Ratio (Strength)", range=x_range, gridcolor='lightgray'),
        yaxis=dict(title="RS Momentum (Trend)", range=y_range, gridcolor='lightgray'),
        height=800,
        template="plotly_white",
        title=f"Relative Rotation Graph as of {rrg_date}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Add Annotations for Quadrants in corners using paper coordinates for fixed position
    fig_rrg.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text="LEADING", showarrow=False, font=dict(size=25, color="green"), opacity=0.3, align="right")
    fig_rrg.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text="IMPROVING", showarrow=False, font=dict(size=25, color="blue"), opacity=0.3, align="left")
    fig_rrg.add_annotation(xref="paper", yref="paper", x=0.02, y=0.02, text="LAGGING", showarrow=False, font=dict(size=25, color="red"), opacity=0.3, align="left")
    fig_rrg.add_annotation(xref="paper", yref="paper", x=0.98, y=0.02, text="WEAKENING", showarrow=False, font=dict(size=25, color="orange"), opacity=0.3, align="right")

    st.plotly_chart(fig_rrg, use_container_width=True)

with tab3:
    st.subheader("Financial Metrics & Advanced Greeks")
    # Dynamic calculation for the selected range
    raw_prices = df_price.loc[start_date:end_date]
    metrics_df = calculate_metrics(raw_prices)
    
    st.dataframe(metrics_df, height=500, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Correlation Matrix (Filtered Range)**")
        corr = raw_prices.pct_change().corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.write("**Risk-Return Profile (Annualized)**")
        plot_data = metrics_df.copy()
        plot_data['CAGR_val'] = plot_data['CAGR'].str.rstrip('%').astype(float)
        plot_data['Vol_val'] = plot_data['Volatility'].str.rstrip('%').astype(float)
        fig_risk = px.scatter(plot_data, x="Vol_val", y="CAGR_val", text=plot_data.index,
                             labels={"Vol_val": "Annual Volatility (%)", "CAGR_val": "Annualized Return (%)"},
                             height=500)
        fig_risk.update_traces(textposition='top center')
        st.plotly_chart(fig_risk, use_container_width=True)

with tab4:
    st.subheader("Sector Seasonality Analysis")
    sector_choice = st.selectbox("Select Sector for Monthly Heatmap", options=df_price.columns.tolist())
    
    m_file = f"analysis_results/monthly_heatmaps/{sector_choice.lower().replace(' ', '_')}.csv"
    if os.path.exists(m_file):
        m_df = pd.read_csv(m_file, index_col=0)
        m_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Use robust color scaling (clip at 2nd and 98th percentile for visualization)
        z_values = m_df.values.flatten()
        z_min, z_max = np.nanpercentile(z_values, [2, 98])
        
        fig_hm = px.imshow(m_df, text_auto=".1f", color_continuous_scale='RdYlGn', aspect="auto",
                          title=f"Monthly Returns for {sector_choice}", 
                          zmin=z_min, zmax=z_max, height=600)
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning(f"Heatmap data for {sector_choice} not found. Please run analysis.py first.")

    st.divider()
    st.subheader("Day-of-the-Week Seasonality")
    weekday_df = pd.read_csv('analysis_results/weekday_analysis.csv', index_col=0)
    
    # Robust scale for weekday heatmap
    w_values = weekday_df.values.flatten()
    w_min, w_max = np.nanpercentile(w_values, [5, 95])
    
    fig_week = px.imshow(weekday_df.T, text_auto=".3f", color_continuous_scale='RdYlGn',
                        labels={"x": "Weekday", "y": "Sector"}, 
                        title="Average Daily Returns by Weekday",
                        zmin=w_min, zmax=w_max, height=800)
    st.plotly_chart(fig_week, use_container_width=True)

st.sidebar.info("""
**RRG Quadrants:**
- **Leading**: Strong Relative Strength & Positive Momentum
- **Weakening**: Strong Strength, but Momentum is declining
- **Lagging**: Weak Strength & Negative Momentum
- **Improving**: Weak Strength, but Momentum is gaining
""")
