import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nifty Sector Rotation | Bloom Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main Background & Font */
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
    }
    label[data-testid="stMetricLabel"] {
        color: #8b949e;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #21262d;
        border-radius: 4px;
        color: #c9d1d9;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
        border-color: #238636 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar {
        orientation: v;
        top: 0;
        right: -30px; 
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
@st.cache_data
def load_data():
    """Loads and cleans price data with robust glitch removal."""
    try:
        df = pd.read_csv('data/combined_sectoral_data.csv', index_col=0, parse_dates=True)
        
        # Robust Data Cleansing (Ratio-to-Benchmark)
        benchmark_col = 'Nifty 50'
        if benchmark_col in df.columns:
            bench_temp = df[benchmark_col].interpolate(limit_direction='both')
            for col in df.columns:
                if col == benchmark_col: continue
                col_temp = df[col].interpolate(limit_direction='both')
                ratio = col_temp / bench_temp
                # 252-day window catches glitches
                median_ratio = ratio.rolling(window=252, center=True, min_periods=1).median()
                outliers = ((ratio - median_ratio).abs() / median_ratio) > 0.30
                df.loc[outliers, col] = np.nan
        
        # Bridge gaps
        df = df.interpolate(method='linear', limit_area='inside').ffill().bfill()
        return df
    except FileNotFoundError:
        st.error("Data file 'data/combined_sectoral_data.csv' not found. Please run analysis.py.")
        return pd.DataFrame()

@st.cache_data
def load_rrg_data():
    """Loads pre-calculated RRG data."""
    try:
        rs_ratio = pd.read_csv('analysis_results/rs_ratio.csv', index_col=0, parse_dates=True)
        rs_momentum = pd.read_csv('analysis_results/rs_momentum.csv', index_col=0, parse_dates=True)
        rs_ratio = rs_ratio.interpolate(method='linear').ffill().bfill()
        rs_momentum = rs_momentum.interpolate(method='linear').ffill().bfill()
        return rs_ratio, rs_momentum
    except FileNotFoundError:
        st.error("RRG analysis files not found in 'analysis_results/'. Please run analysis.py.")
        return pd.DataFrame(), pd.DataFrame()

def calculate_stats(prices, benchmark_col='Nifty 50'):
    """Calculates detailed financial metrics."""
    returns = prices.pct_change().dropna(how='all')
    if benchmark_col not in returns.columns:
        return pd.DataFrame()

    bench_returns = returns[benchmark_col].dropna()
    results = {}
    
    for col in prices.columns:
        col_prices = prices[col].dropna()
        if len(col_prices) < 2: continue
            
        # Return Stats
        total_ret = (col_prices.iloc[-1] / col_prices.iloc[0]) - 1
        days = (col_prices.index[-1] - col_prices.index[0]).days
        years = days / 365.25
        cagr = (1 + total_ret)**(1/years) - 1 if years > 0 else 0
        
        # Risk Stats
        col_rets = col_prices.pct_change().dropna()
        if len(col_rets) == 0: continue
            
        vol = col_rets.std() * np.sqrt(252)
        sharpe = (cagr - 0.06) / vol if vol > 0 else 0
        
        # Drawdown
        cum = (1 + col_rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        # Alpha/Beta
        aligned = pd.concat([bench_returns, col_rets], axis=1).dropna()
        if len(aligned) > 20:
            slope, intercept, _, _, _ = stats.linregress(aligned.iloc[:,0], aligned.iloc[:,1])
            beta = slope
            alpha = intercept * 252
        else:
            beta, alpha = np.nan, np.nan
            
        results[col] = {
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "Beta": beta,
            "Alpha": alpha
        }
    return pd.DataFrame(results).T

# --- LOAD RESOURCES ---
df_price = load_data()
rs_ratio, rs_momentum = load_rrg_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    # Date Range Filter
    min_date = df_price.index.min()
    max_date = df_price.index.max()
    
    st.subheader("Time Period")
    date_range = st.date_input(
        "Select Analysis Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    st.subheader("Sector Universe")
    available_sectors = df_price.columns.tolist() if not df_price.empty else []
    selected_sectors = st.multiselect("Select Sectors to Compare", options=available_sectors, default=available_sectors[:5])

    st.markdown("---")
    st.info("**Tip:** Use the 'Metrics' tab to see detailed risk-adjusted returns for your selected period.")

# --- FILTER DATA ---
if not df_price.empty:
    filtered_df = df_price.loc[start_date:end_date]
    if not filtered_df.empty:
        # Rebase to 100
        rebased_df = filtered_df.copy()
        for col in rebased_df.columns:
            first_val = rebased_df[col].dropna().iloc[0] if not rebased_df[col].dropna().empty else np.nan
            rebased_df[col] = (rebased_df[col] / first_val) * 100

# --- MAIN CONTENT ---
st.title("🇮🇳 Market Compass: Sector Rotation")
st.markdown("Professional analysis of Nifty Sectoral Indices using Relative Rotation Graphs (RRG) and Advanced Greeks.")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 RRG Analysis", "📈 Performance", "📊 Deep Dive Metrics", "🗓️ Seasonality"])

# TAB 1: RRG
with tab1:
    if rs_ratio.empty:
        st.warning("No RRG data available.")
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("### 🛠 Time Machine")
            # Snapshot Slider
            all_dates = rs_ratio.index.strftime('%Y-%m-%d').tolist()
            curr_date_str = st.select_slider(
                "Snapshot Date", 
                options=all_dates, 
                value=all_dates[-1],
                label_visibility="collapsed"
            )
            trail_len = st.slider("Trail Length (Weeks)", 1, 52, 12)
            
            st.markdown("---")
            st.markdown("""
            **Quadrants:**
            - 🟢 **Leading**: Strong Trend
            - 🔵 **Improving**: Turnaround
            - 🔴 **Lagging**: Underperformance
            - 🟠 **Weakening**: Profit Taking
            """)
            
        with col2:
            snap_idx = rs_ratio.index.get_loc(curr_date_str)
            start_trail = max(0, snap_idx - trail_len)
            
            # Slice Data
            r_slice = rs_ratio.iloc[start_trail:snap_idx+1]
            m_slice = rs_momentum.iloc[start_trail:snap_idx+1]
            
            fig_rrg = go.Figure()
            
            # Quadrant Backgrounds
            fig_rrg.add_shape(type="rect", x0=100, x1=200, y0=100, y1=200, fillcolor="rgba(0, 128, 0, 0.05)", line_width=0, layer="below")
            fig_rrg.add_shape(type="rect", x0=0, x1=100, y0=100, y1=200, fillcolor="rgba(0, 0, 255, 0.05)", line_width=0, layer="below")
            fig_rrg.add_shape(type="rect", x0=0, x1=100, y0=0, y1=100, fillcolor="rgba(255, 0, 0, 0.05)", line_width=0, layer="below")
            fig_rrg.add_shape(type="rect", x0=100, x1=200, y0=0, y1=100, fillcolor="rgba(255, 165, 0, 0.05)", line_width=0, layer="below")
            
            # Axes
            fig_rrg.add_vline(x=100, line_width=1, line_color="#888")
            fig_rrg.add_hline(y=100, line_width=1, line_color="#888")
            
            # Plot Trails
            for col in r_slice.columns:
                fig_rrg.add_trace(go.Scatter(
                    x=r_slice[col], 
                    y=m_slice[col],
                    mode='lines+markers',
                    name=col,
                    marker=dict(size=[3]*(len(r_slice)-1) + [12], opacity=[0.3]*(len(r_slice)-1) + [1]),
                    line=dict(width=2),
                    hovertemplate=f"<b>{col}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}"
                ))

            # Dynamic Range
            xmax, xmin = r_slice.max().max(), r_slice.min().min()
            ymax, ymin = m_slice.max().max(), m_slice.min().min()
            
            fig_rrg.update_layout(
                title=dict(text=f"Relative Rotation Graph ({curr_date_str})", x=0.02, y=0.98),
                xaxis=dict(title="Relative Strength (vs Nifty 50)", range=[xmin-1, xmax+1], gridcolor='#30363d'),
                yaxis=dict(title="Momentum", range=[ymin-1, ymax+1], gridcolor='#30363d'),
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                showlegend=True,
                legend=dict(orientation="h", x=0, y=1.05)
            )
            
            # Text Annotations
            fig_rrg.add_annotation(x=101, y=101, text="LEADING", showarrow=False, font=dict(size=30, color="rgba(0,128,0,0.1)"))
            fig_rrg.add_annotation(x=99, y=101, text="IMPROVING", showarrow=False, font=dict(size=30, color="rgba(0,0,255,0.1)"))
            fig_rrg.add_annotation(x=99, y=99, text="LAGGING", showarrow=False, font=dict(size=30, color="rgba(255,0,0,0.1)"))
            fig_rrg.add_annotation(x=101, y=99, text="WEAKENING", showarrow=False, font=dict(size=30, color="rgba(255,165,0,0.1)"))
            
            st.plotly_chart(fig_rrg, use_container_width=True)

# TAB 2: PERFORMANCE
with tab1: # Actually user wanted tab2 to be performance
    pass 
with tab2:
    st.subheader("📊 Comparative Performance (Rebased)")
    if not selected_sectors:
        st.info("Please select sectors in the sidebar to compare.")
    else:
        fig_perf = px.line(
            rebased_df[selected_sectors], 
            labels={"value": "Performance Index (Base=100)", "variable": "Sector"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_perf.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            hovermode="x unified",
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d')
        )
        st.plotly_chart(fig_perf, use_container_width=True)

# TAB 3: METRICS
with tab3:
    st.subheader("🧮 Financial Statistics")
    
    if filtered_df.empty:
        st.write("No data in range.")
    else:
        metrics = calculate_stats(filtered_df.loc[start_date:end_date])
        if not metrics.empty:
            # Formatting
            fmt_metrics = metrics.copy()
            fmt_metrics['CAGR'] = fmt_metrics['CAGR'].apply(lambda x: f"{x:.2%}")
            fmt_metrics['Volatility'] = fmt_metrics['Volatility'].apply(lambda x: f"{x:.2%}")
            fmt_metrics['Sharpe'] = fmt_metrics['Sharpe'].apply(lambda x: f"{x:.2f}")
            fmt_metrics['MaxDD'] = fmt_metrics['MaxDD'].apply(lambda x: f"{x:.2%}")
            fmt_metrics['Alpha'] = fmt_metrics['Alpha'].apply(lambda x: f"{x:.2%}")
            fmt_metrics['Beta'] = fmt_metrics['Beta'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(fmt_metrics, use_container_width=True, height=500)
            
            # Risk/Return Scatter
            st.markdown("#### Risk vs. Return Landscape")
            scatter_df = metrics.copy()
            fig_risk = px.scatter(
                scatter_df, x="Volatility", y="CAGR", text=scatter_df.index,
                size=[15]*len(scatter_df), color="Sharpe",
                color_continuous_scale="RdYlGn",
                labels={"Volatility": "Annualized Volatility (Risk)", "CAGR": "Annualized Return"},
                title="Efficient Frontier Visualization"
            )
            fig_risk.update_traces(textposition='top center')
            fig_risk.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                xaxis=dict(gridcolor='#30363d'),
                yaxis=dict(gridcolor='#30363d')
            )
            st.plotly_chart(fig_risk, use_container_width=True)

# TAB 4: SEASONALITY
with tab4:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("📅 Monthly Patterns")
        seat_sec = st.selectbox("Select Sector", options=available_sectors)
    
    with col2:
        m_file = f"analysis_results/monthly_heatmaps/{seat_sec.lower().replace(' ', '_')}.csv"
        if os.path.exists(m_file):
            m_df = pd.read_csv(m_file, index_col=0)
            m_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Robust scaling
            vals = m_df.values.flatten()
            vmin, vmax = np.nanpercentile(vals, [5, 95])
            
            fig_hm = px.imshow(
                m_df, 
                text_auto=".1f", 
                color_continuous_scale='RdYlGn', 
                aspect="auto",
                zmin=vmin, zmax=vmax
            )
            fig_hm.update_layout(
                title=f"Monthly Returns Heatmap: {seat_sec}",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa')
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning(f"No seasonality data found for {seat_sec}")
    
    st.markdown("---")
    st.subheader("📆 Day-of-Week Effect")
    if os.path.exists('analysis_results/weekday_analysis.csv'):
        w_df = pd.read_csv('analysis_results/weekday_analysis.csv', index_col=0)
        fig_wk = px.imshow(
            w_df.T, 
            text_auto=".2f", 
            color_continuous_scale='RdYlGn',
            height=600
        )
        fig_wk.update_layout(
            title="Average Daily Return by Weekday",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig_wk, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Based on data from Nifty Indices. Built with ❤️ and Python.")
