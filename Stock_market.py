import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import os

# =========================
# 1. PAGE CONFIG & THEME
# =========================
st.set_page_config(page_title="BTC Volatility Intelligence", page_icon="📊", layout="wide")

st.title("₿ BTC Volatility Intelligence Dashboard")
st.markdown("### Advanced Financial Analytics & Market Regime Detection")

# =========================
# 2. DATA ENGINE (AUTO-DETECT)
# =========================
@st.cache_data
def load_market_data():
    """Searches directory for the dataset and cleans it for analysis."""
    target_keywords = ["btcusd", "1-min", "data"]
    found_file = None
    
    # Auto-detect file in the current directory
    for file in os.listdir("."):
        if any(key in file.lower() for key in target_keywords):
            found_file = file
            break
            
    if not found_file:
        return None, None

    try:
        with open(found_file, 'r', encoding='utf-8') as f:
            # Handle the specific CSV formatting (removing outer quotes)
            clean_lines = [line.strip().strip('"') for line in f if line.strip()]
        
        if not clean_lines:
            return None, found_file
            
        df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
        df.columns = df.columns.str.strip().str.lower()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')

        return df.dropna(subset=['close', 'timestamp']), found_file
    except:
        return None, found_file

# =========================
# 3. SIDEBAR CONTROLS
# =========================
st.sidebar.header("Dashboard Controls")

df_raw, detected_name = load_market_data()

if df_raw is not None:
    st.sidebar.success(f"Data Source: {detected_name}")
    analysis_mode = st.sidebar.radio("Data Source:", ["Live Historical Data", "Synthetic Simulation"])
else:
    st.sidebar.warning("No historical file detected. Running in Simulation Mode.")
    analysis_mode = "Synthetic Simulation"

rolling_window = st.sidebar.slider("Rolling Window (Minutes)", 10, 500, 60)
percentile_threshold = st.sidebar.slider("Volatility Sensitivity (%)", 70, 99, 85)

# =========================
# 4. ANALYTICS LOGIC
# =========================
if analysis_mode == "Live Historical Data" and df_raw is not None:
    data_percent = st.sidebar.slider("Historical Depth (%)", 5, 100, 25)
    limit = int(len(df_raw) * (data_percent / 100))
    df_final = df_raw.iloc[:limit].copy()
else:
    # Synthetic Model using Sine waves, Stochastic Noise, and Drift
    t = np.linspace(0, 100, 2000)
    sine = 15 * np.sin(t / 5) 
    noise = np.random.normal(0, 5, 2000)
    drift = np.cumsum(np.random.normal(0.1, 0.4, 2000))
    df_final = pd.DataFrame({
        'timestamp': pd.date_range(start="2024-01-01", periods=2000, freq='min'),
        'close': 100 + sine + noise + drift
    })

# Compute Volatility Metrics
df_final["rolling_std"] = df_final["close"].rolling(window=rolling_window).std()
df_view = df_final.tail(10000).copy()

# Dynamic Regime Classification
valid_std = df_view["rolling_std"].dropna()
thresh = np.percentile(valid_std, percentile_threshold) if not valid_std.empty else 0
df_view["zone"] = np.where(df_view["rolling_std"] > thresh, "Volatile", "Stable")

# =========================
# 5. DASHBOARD VISUALS
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("Active Mode", analysis_mode)
m2.metric("Volatility Threshold", f"{thresh:.2f}")
m3.metric("Current Regime", df_view["zone"].iloc[-1])

st.subheader("Price Action & Regime Classification")
fig = go.Figure()
for zone, color in [("Stable Mode", "#00FF00"), ("Volatile Mode", "#FF0000")]:
    mask = df_view["zone"] == (zone.split(" ")[0])
    fig.add_trace(go.Scatter(
        x=df_view.loc[mask, "timestamp"], 
        y=df_view.loc[mask, "close"],
        mode="markers", 
        name=zone, 
        marker=dict(color=color, size=2.5)
    ))
fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("**Volatility Intensity (Rolling Std Dev)**")
    fig_vol = px.line(df_view, x="timestamp", y="rolling_std", template="plotly_dark")
    fig_vol.add_hline(y=thresh, line_dash="dash", line_color="yellow", annotation_text="Regime Break")
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    st.write("**Price Distribution Analysis**")
    fig_hist = px.histogram(df_view, x="close", nbins=50, template="plotly_dark", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# 6. TECHNICAL DOCUMENTATION
# =========================
with st.expander("Quantitative Methodology"):
    st.markdown("### Methodology Overview")
    st.write("This dashboard utilizes quantitative finance techniques to analyze price behavior:")
    st.write("- **Stochastic Modeling:** Market noise is identified using Gaussian distribution variance.")
    st.write("- **Dynamic Thresholding:** Regimes are classified using percentile-based rolling standard deviation.")
    st.write("- **Synthetic Simulation:** Uses periodic functions and discrete integrals to model price drift.")
    st.latex(r"\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}")

# Data Export
csv = df_view.to_csv(index=False).encode('utf-8')
st.sidebar.divider()
st.sidebar.download_button("📥 Export Analysis CSV", data=csv, file_name="market_analysis.csv")
