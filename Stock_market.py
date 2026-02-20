import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import io

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BTC Volatility Intelligence",
    layout="wide"
)

st.title(" BTC 1-Min Volatility Intelligence Dashboard")
st.markdown("Mathematics for AI | Advanced Financial Analytics")

# =========================
# FILE PATH
# =========================
FILE_PATH = "btcusd_1-min_data.csv.crdownload"

if not os.path.exists(FILE_PATH):
    st.error(f"Dataset not found at {FILE_PATH}.")
    st.stop()

# =========================
# CLEAN & CACHED DATA LOADING
# =========================
@st.cache_data
def load_data(path):
    """Loads and cleans the raw file once."""
    with open(path, 'r') as f:
        # Strip whitespace and surrounding quotes from every non-empty line
        clean_lines = [line.strip().strip('"') for line in f if line.strip()]
    
    df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert timestamp (dataset uses UNIX timestamp)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce')

    # Drop rows with critical missing values
    df = df.dropna(subset=['close', 'timestamp'])
    return df

# Initial load of the full dataset
with st.spinner("Loading full dataset into cache..."):
    df_raw = load_data(FILE_PATH)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header(" Global Data Settings")

# NEW: Control the total amount of data used from the CSV
total_rows = len(df_raw)
data_percent = st.sidebar.slider(
    "Total Dataset Usage (%)", 
    min_value=5, 
    max_value=100, 
    value=50,
    help="Select what percentage of the 637k rows to include in calculations."
)

# Calculate slice based on percentage
slice_limit = int(total_rows * (data_percent / 100))
df_sliced = df_raw.iloc[:slice_limit].copy()

st.sidebar.divider()
st.sidebar.header(" Visualization Controls")

# Slider for the "View" (The recent points to show on charts)
sample_size = st.sidebar.slider(
    "Points to Chart (Tail of slice)",
    min_value=500,
    max_value=min(len(df_sliced), 50000),
    value=min(len(df_sliced), 15000),
    step=500
)

rolling_window = st.sidebar.slider("Rolling Window (Minutes)", 10, 500, 60)
percentile_threshold = st.sidebar.slider("Volatility Percentile %", 70, 99, 85)

# =========================
# ANALYSIS LOGIC
# =========================
# We calculate volatility on the SLICED data first so calculations are consistent
df_sliced["rolling_std"] = df_sliced["close"].rolling(window=rolling_window).std()

# Get the "View" for the dashboard
df_view = df_sliced.tail(sample_size).copy()

# Dynamic Threshold
valid_std = df_view["rolling_std"].dropna()
if not valid_std.empty:
    threshold_value = np.percentile(valid_std, percentile_threshold)
else:
    threshold_value = 0

df_view["zone"] = np.where(
    df_view["rolling_std"] > threshold_value, 
    "Volatile", 
    "Stable"
)

# =========================
# METRICS SECTION
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows in Use", f"{len(df_sliced):,}")
col2.metric("Avg Price", f"${df_view['close'].mean():,.2f}")
col3.metric("Max Price", f"${df_view['close'].max():,.2f}")
col4.metric("Volatility Threshold", f"{threshold_value:.2f}")

# =========================
# CHARTS
# =========================
st.subheader(f" Price Movement ({sample_size:,} Latest Points)")

fig = go.Figure()
for zone, color in [("Stable", "#00FF00"), ("Volatile", "#FF0000")]:
    mask = df_view["zone"] == zone
    fig.add_trace(go.Scatter(
        x=df_view.loc[mask, "timestamp"],
        y=df_view.loc[mask, "close"],
        mode="markers",
        name=zone,
        marker=dict(color=color, size=2)
    ))

fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="BTC Price (USD)")
st.plotly_chart(fig, use_container_width=True)

st.subheader(" Volatility Intensity (Rolling Std Dev)")
fig_vol = px.line(df_view, x="timestamp", y="rolling_std", template="plotly_dark")
fig_vol.add_hline(y=threshold_value, line_dash="dash", line_color="yellow")
st.plotly_chart(fig_vol, use_container_width=True)

# =========================
# DOWNLOAD & INFO
# =========================
st.sidebar.divider()
st.sidebar.write(f"**Total Available:** {total_rows:,} rows")
st.sidebar.write(f"**Currently Processing:** {len(df_sliced):,} rows")

csv = df_view[["timestamp", "close", "rolling_std", "zone"]].to_csv(index=False).encode("utf-8")
st.download_button("Download Current View CSV", data=csv, file_name="btc_analysis.csv")

st.markdown(f"""
### Dataset Insights:
* You are currently analyzing **{data_percent}%** of the historical dataset.
* The volatility threshold is calculated using the **{percentile_threshold}th percentile** of the current view.
* Increasing the "Total Dataset Usage" slider will allow you to look further back into the history of the file.
""")
