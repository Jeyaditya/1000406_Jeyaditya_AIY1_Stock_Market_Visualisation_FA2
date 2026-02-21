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
# 2. ROBUST DATA LOADER
# =========================
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Important: strip() removes whitespace, strip('"') removes the outer quotes
            clean_lines = [line.strip().strip('"') for line in f if line.strip()]
        
        if not clean_lines:
            return None
            
        df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
        df.columns = df.columns.str.strip().str.lower()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')

        return df.dropna(subset=['close', 'timestamp'])
    except:
        return None

# =========================
# 3. SIDEBAR CONTROLS
# =========================
st.sidebar.header("🕹️ Dashboard Controls")

mode = st.sidebar.radio(
    "Analysis Mode:", 
    ["Real BTC Data", "Math Logic Simulation"],
    help="Simulation mode uses sine waves and integrals as per rubric requirements."
)

rolling_window = st.sidebar.slider("Rolling Window (Minutes)", 10, 500, 60)
percentile_threshold = st.sidebar.slider("Volatility Percentile", 70, 99, 85)

# =========================
# 4. DATA PROCESSING
# =========================
df_final = None

if mode == "Real BTC Data":
    FILE_PATH = "btcusd_1-min_data.csv.crdownload"
    df_raw = load_data(FILE_PATH)
    
    if df_raw is not None:
        data_percent = st.sidebar.slider("Dataset Usage (%)", 5, 100, 50)
        limit = int(len(df_raw) * (data_percent / 100))
        df_final = df_raw.iloc[:limit].copy()
    else:
        st.warning("Dataset file not found. Switching to Simulation Mode...")
        mode = "Math Logic Simulation"

if mode == "Math Logic Simulation" or df_final is None:
    # Rubric Requirement: Use Sine/Noise/Integrals
    t = np.linspace(0, 100, 2000)
    sine = 15 * np.sin(t / 5) 
    noise = np.random.normal(0, 5, 2000)
    drift = np.cumsum(np.random.normal(0.1, 0.4, 2000))
    
    price = 100 + sine + noise + drift
    df_final = pd.DataFrame({
        'timestamp': pd.date_range(start="2024-01-01", periods=2000, freq='min'),
        'close': price
    })

# Calculations
df_final["rolling_std"] = df_final["close"].rolling(window=rolling_window).std()
df_view = df_final.tail(10000).copy()

valid_std = df_view["rolling_std"].dropna()
thresh = np.percentile(valid_std, percentile_threshold) if not valid_std.empty else 0
df_view["zone"] = np.where(df_view["rolling_std"] > thresh, "Volatile", "Stable")

# =========================
# 5. VISUALS
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("Mode", mode)
m2.metric("Threshold", f"{thresh:.2f}")
m3.metric("Status", df_view["zone"].iloc[-1])

st.subheader("📊 Price Action & Regime Detection")
fig = go.Figure()
for zone, color in [("Stable", "#00FF00"), ("Volatile", "#FF0000")]:
    mask = df_view["zone"] == zone
    fig.add_trace(go.Scatter(x=df_view.loc[mask, "timestamp"], y=df_view.loc[mask, "close"],
                             mode="markers", name=zone, marker=dict(color=color, size=2)))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("**Volatility Intensity (Rolling Std Dev)**")
    fig_vol = px.line(df_view, x="timestamp", y="rolling_std", template="plotly_dark")
    fig_vol.add_hline(y=thresh, line_dash="dash", line_color="yellow")
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    st.write("**Distribution of Price Noise**")
    fig_hist = px.histogram(df_view, x="close", nbins=50, template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)


# =========================
# 6. RUBRIC: INSIGHTFUL EXPLANATIONS
# =========================
with st.expander("📚 Mathematical Rationale"):
    # Fixed the f-string syntax error by using standard markdown
    st.markdown("### Logic & AI Framework")
    st.write("- **Stochastic Noise:** Modeled via Gaussian distributions.")
    st.write("- **Periodic Swings:** Simulated using Sine functions.")
    st.write("- **Trend (Drift):** Modeled using discrete integrals.")
    st.latex(r"\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}")

# Download Button
csv = df_view.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Analysis CSV", data=csv, file_name="analysis.csv")
