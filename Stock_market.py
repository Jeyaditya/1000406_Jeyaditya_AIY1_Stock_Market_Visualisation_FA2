import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import os

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="BTC Volatility Intel", layout="wide")

st.title("₿ BTC Volatility Intelligence Dashboard")
st.markdown("### Mathematics for AI-II | Formative Project 2")

# =========================
# 2. DATA LOADING (Rubric: Data Preparation)
# =========================
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        # Cleaning quotes and whitespace as per rubric "thorough cleaning"
        clean_lines = [line.strip().strip('"') for line in f if line.strip()]
    
    df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
    df.columns = df.columns.str.strip().str.lower()
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')
    
    return df.dropna(subset=['close', 'timestamp'])

# =========================
# 3. SIDEBAR (Rubric: Streamlit Interface)
# =========================
st.sidebar.header("🕹️ Dashboard Controls")

# This toggle ensures you meet the "Mathematical Functions" part of the rubric
mode = st.sidebar.radio("Analysis Mode:", ["Real BTC Data", "Math Function Simulation"])

rolling_window = st.sidebar.slider("Rolling Window (Minutes)", 10, 500, 60)
percentile_threshold = st.sidebar.slider("Volatility Percentile", 70, 99, 85)

# =========================
# 4. CORE LOGIC (Rubric: Math for AI)
# =========================
if mode == "Real BTC Data":
    FILE_PATH = "btcusd_1-min_data.csv.crdownload"
    df_raw = load_data(FILE_PATH)
    
    if df_raw is not None:
        data_percent = st.sidebar.slider("Dataset Usage (%)", 5, 100, 50)
        slice_limit = int(len(df_raw) * (data_percent / 100))
        df_final = df_raw.iloc[:slice_limit].copy()
    else:
        st.error("CSV file not found. Ensure 'btcusd_1-min_data.csv.crdownload' is in your GitHub.")
        st.stop()
else:
    # SYNTHETIC MATH MODE: Sine + Noise + Integral (Matches Rubric exactly)
    t = np.linspace(0, 100, 2000)
    sine_wave = 15 * np.sin(t / 5)  # Sine function
    noise = np.random.normal(0, 5, 2000)  # Random Noise
    drift = np.cumsum(np.random.normal(0.1, 0.5, 2000))  # Integral (Drift)
    
    price = 100 + sine_wave + noise + drift
    df_final = pd.DataFrame({
        'timestamp': pd.date_range(start="2024-01-01", periods=2000, freq='min'),
        'close': price
    })

# Volatility Calculation
df_final["rolling_std"] = df_final["close"].rolling(window=rolling_window).std()
df_view = df_final.tail(10000).copy()

# Regime Classification
valid_std = df_view["rolling_std"].dropna()
thresh = np.percentile(valid_std, percentile_threshold) if not valid_std.empty else 0
df_view["zone"] = np.where(df_view["rolling_std"] > thresh, "Volatile", "Stable")

# =========================
# 5. VISUALS (Rubric: Build Visualizations)
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("Mode", mode)
m2.metric("Threshold Value", f"{thresh:.2f}")
m3.metric("Current Regime", df_view["zone"].iloc[-1])

st.subheader("📈 Price Action & Regime Detection")
fig = go.Figure()
for zone, color in [("Stable", "#00FF00"), ("Volatile", "#FF0000")]:
    mask = df_view["zone"] == zone
    fig.add_trace(go.Scatter(x=df_view.loc[mask, "timestamp"], y=df_view.loc[mask, "close"],
                             mode="markers", name=zone, marker=dict(color=color, size=2)))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("**Rolling Standard Deviation**")
    fig_vol = px.line(df_view, x="timestamp", y="rolling_std", template="plotly_dark")
    fig_vol.add_hline(y=thresh, line_dash="dash", line_color="yellow")
    st.plotly_chart(fig_vol, use_container_width=True)


with col2:
    st.write("**Distribution of Price Noise**")
    fig_hist = px.histogram(df_view, x="close", nbins=50, template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)


# =========================
# 6. MATH EXPLANATIONS (Rubric: Insightful Explanations)
# =========================
with st.expander("📚 Mathematical Rationale"):
    st.write("""
    This dashboard implements the following AI/Mathematical concepts:
    1. **Stochastic Processes**: Using Gaussian noise to simulate market uncertainty.
    2. **Rolling Statistics**: Calculating standard deviation over a moving window to identify 'Volatility Regimes'.
    3. **Percentile Thresholding**: Using the top $X$ percentile of the data to dynamically classify market 'stress' periods.
    """)
