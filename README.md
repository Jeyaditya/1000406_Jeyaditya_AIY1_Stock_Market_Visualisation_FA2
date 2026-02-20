# 1000406_Jeyaditya_AIY1_Stock_Market_Visualisation_FA2
Crypto Volatility visualizer appBTC Volatility Intelligence Dashboard
Mathematics for AI | Advanced Financial Analytics

An interactive Streamlit-based intelligence dashboard designed to analyze and visualize Bitcoin (BTC) market volatility. Using rolling standard deviation and percentile-based thresholding, this tool identifies "Stable" vs "Volatile" market zones across historical 1-minute data.
# Features:

  * Dynamic Data Slicing: Process anywhere from 5% to 100% of the massive 637k+ row dataset.

  * Mathematical Volatility Detection: Uses a Rolling Standard Deviation window (adjustable via UI).

  *  Adaptive Thresholding: Automatically classifies market zones using percentile-based logic.

  * Interactive Visuals: Built with Plotly for high-performance zooming and inspection of price action.

  *Fail-Proof Parsing: Custom data-cleaning logic to handle quoted CSV strings and UNIX timestamps.

# Mathematical Approach

The dashboard utilizes the Rolling Standard Deviation formula to measure market "stress":
σ=N−11​i=1∑N​(xi​−xˉ)2​

Where:

  * N is the Rolling Window selected in the sidebar.

  * xi​ is the Bitcoin Close price at a given minute.

  * xˉ is the mean price over that window.

Points exceeding the selected Percentile Threshold (e.g., 85th percentile) are categorized as Volatile (Red), while others remain Stable (Green).
# Installation & Setup

    Clone the repository:
    Bash

    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME

    Install dependencies:
    Bash

    pip install streamlit pandas numpy plotly

    Prepare the dataset:
    Ensure your dataset is located at: images/btcusd_1-min_data.csv

    Run the application:
    Bash

    streamlit run Stock_market.py

# Usage

  * Total Dataset Usage: Use the top slider to determine how much historical data to load into memory.
  * Rolling Window: Increase for a "macro" view of volatility; decrease for "micro" noise detection.
  * Volatility Percentile: Adjust how sensitive the dashboard is to price swings.
  * Download: Export the analyzed data (including the volatility zones) as a CSV for further research.
