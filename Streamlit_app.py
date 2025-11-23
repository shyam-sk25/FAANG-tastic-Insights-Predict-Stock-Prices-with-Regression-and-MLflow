import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow
import plotly.express as px

# --- Utility Function for Formatting Large Numbers ---
def format_large_number(num):
    """
    Formats a large number into a readable string with M (Million), B (Billion), 
    or T (Trillion) suffix, suitable for US financial data.
    """
    num = float(num) # Ensure the input is treated as a float
    if abs(num) >= 1e12:
        return f'${num / 1e12:,.2f}T'
    elif abs(num) >= 1e9:
        return f'${num / 1e9:,.2f}B'
    elif abs(num) >= 1e6:
        # Volume is often displayed without a dollar sign, keep M for million
        if num == abs(num): 
             return f'{num / 1e6:,.2f}M' 
        return f'${num / 1e6:,.2f}M'
    else:
        return f'${num:,.0f}'
# ---------------------------------------------------

# ============================
# 1. PAGE CONFIG
# ============================
st.set_page_config(
    page_title="FAANG Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================
# 2. CUSTOM DARK THEME STYLES (Check this block carefully when pasting!)
# ============================
st.markdown("""
<style>
.stApp { background-color: #1a1a2e; color: #f0f2f6; }
h1, h2, h3, h4 { color: #00ffaa !important; }
[data-testid="stMetricValue"] { color: #00ffaa !important; font-weight: bold; }
[data-testid="stMetricDelta"] { color: #a0a0ff !important; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ============================
# 3. LOAD MODEL ARTIFACTS
# ============================
try:
    # Load files saved in Step 4 of the notebook
    scaler = joblib.load('minmax_scaler.pkl')
    feature_columns = pd.read_csv('feature_columns.csv')['FeatureName'].tolist()

    # Load the model version created in Step 4
    MODEL_NAME = "FAANG_XGBoost_Price_Predictor"
    model_uri = f"models:/{MODEL_NAME}/latest" 
    model = mlflow.sklearn.load_model(model_uri)
    LOAD_SUCCESS = True
    # Placeholder for the R-squared from your successful training run
    r_squared = 0.9853 
except Exception as e:
    LOAD_SUCCESS = False
    st.error(f"❌ Failed to load model artifacts. Ensure MLflow server is running and files are present: {e}")
    # Stop execution if artifacts fail to load
    st.stop()


# ============================
# 4. COMPANY LOGOS
# ============================
logos = {
    "AAPL": "https://logos-world.net/wp-content/uploads/2020/04/Apple-Logo.png",
    "AMZN": "https://logos-world.net/wp-content/uploads/2020/04/Amazon-Logo.png",
    "GOOGL": "https://logos-world.net/wp-content/uploads/2020/09/Google-Logo.png",
    "META": "https://logos-world.net/wp-content/uploads/2021/10/Meta-Logo.png",
    "NFLX": "https://logos-world.net/wp-content/uploads/2020/04/Netflix-Logo.png"
}


# ============================
# 5. MAIN UI (INPUTS)
# ============================
st.title("📈 FAANG Next-Day Stock Price Predictor")

col_logo, col_sel = st.columns([1, 3])
with col_sel:
    stock_ticker = st.selectbox("Select a Stock:", ("AAPL", "AMZN", "GOOGL", "META", "NFLX"))

with col_logo:
    st.image(logos[stock_ticker], width=100)

st.markdown("---")


# ============================
# TIME-SERIES INPUTS (Must match Step 3 features)
# ============================
st.subheader("⏳ Time-Series & Technical Inputs")
col1, col2, col_macd = st.columns(3)

# Inputs for Close_Lag1 and SMA_10
close_lag1 = col1.number_input("Previous Day Close ($) (Close_Lag1)", value=150.00, min_value=1.0, format="%.2f")
sma_10 = col2.number_input("10-Day SMA ($)", value=148.00, min_value=1.0, format="%.2f")

# Inputs for new technical features
rsi = col_macd.number_input("RSI (14-Day)", value=55.0, min_value=0.0, max_value=100.0, format="%.2f")
macd = col_macd.number_input("MACD", value=1.5, format="%.2f")
percent_b = col_macd.number_input("Bollinger %B", value=0.6, format="%.2f")


# ============================
# FUNDAMENTAL INPUTS (UPDATED STEP SIZES)
# ============================
st.subheader("📘 Fundamental & Volume Inputs")
col3, col4, col5 = st.columns(3)

with col3:
    # Changed step to 1 Million (1_000_000)
    volume = st.number_input("Volume (Last Day)", value=100_000_000, min_value=0, step=1_000_000)
    pe_ratio = st.number_input("P/E Ratio", value=30.0, min_value=0.1)
    roe = st.slider("Return on Equity (ROE)", -0.5, 0.5, 0.10)

with col4:
    # Changed step to 1 Billion (1_000_000_000)
    market_cap = st.number_input("Market Cap ($)", value=2_500_000_000_000, step=1_000_000_000)
    beta = st.slider("Beta", 0.0, 3.0, 1.2)
    debt_to_equity = st.slider("Debt-to-Equity", 0.0, 5.0, 1.0)

with col5:
    eps = st.number_input("EPS", value=6.50)
    # Changed step to 1 Billion (1_000_000_000)
    total_debt = st.number_input("Total Debt ($)", value=100_000_000_000, step=1_000_000_000)


# ============================
# PREDICTION SECTION
# ============================
st.markdown("---")

if st.button("🚀 Generate Price Forecast"):

    with st.spinner("Calculating prediction..."):

        # 1. Build Input Row
        input_data = {col: 0.0 for col in feature_columns}

        # Map user inputs to the features
        input_data.update({
            "Volume": volume, 
            "Market_Cap": market_cap, 
            "PE_Ratio": pe_ratio,
            "Beta": beta,
            "EPS": eps,
            "Total_Debt": total_debt, 
            "Return_on_Equity_ROE": roe,
            "Debt_to_Equity": debt_to_equity,
            "Close_Lag1": close_lag1,
            "SMA_10": sma_10,
            "RSI": rsi,
            "MACD": macd,
            "Percent_B": percent_b, 
            "Open": close_lag1, 
            "High": close_lag1 * 1.005, 
            "Low": close_lag1 * 0.995,  
        })
        
        # Map categorical/dummy features
        stock_col = f"Stock_{stock_ticker}"
        if stock_col in input_data:
            input_data[stock_col] = 1.0
        
        analyst_col = "Analyst_buy"
        if analyst_col in input_data:
             input_data[analyst_col] = 1.0

        input_df = pd.DataFrame([input_data])[feature_columns]

        # 2. Scale Numerical Features
        num_cols = [c for c in input_df.columns if input_df[c].dtype in [np.float64, np.int64]]
        cols_to_exclude = [c for c in num_cols if c.startswith(("Stock_", "Analyst_"))]
        numerical_cols_to_scale = [c for c in num_cols if c not in cols_to_exclude]

        input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])

        # 3. Make the Prediction
        prediction = float(model.predict(input_df)[0])
        delta = prediction - close_lag1

        # Use fixed metrics for display
        mae_approx = 11.02 
        lower = prediction - (mae_approx * 1.1)
        upper = prediction + (mae_approx * 1.1)

    st.success("🎉 Forecast generated successfully!")

    st.metric(
        label=f"Predicted Close Price for {stock_ticker}",
        value=f"${prediction:,.2f}",
        delta=f"{delta:,.2f} USD"
    )

    st.write(f"📘 **Confidence Range (±1.1 MAE):** `${lower:,.2f}` - `${upper:,.2f}`")


# ============================
# MODEL INFO (SIDEBAR) - Displays formatted large numbers
# ============================
st.sidebar.header("📄 Model Information")

# --- Display Input Metrics in Readable Format ---
st.sidebar.markdown("---")
st.sidebar.subheader("Current Input Metrics")
st.sidebar.metric("Market Cap", format_large_number(market_cap))
st.sidebar.metric("Total Debt", format_large_number(total_debt))
st.sidebar.metric("Volume (Last Day)", format_large_number(volume).replace('$', '')) 

# --- Existing Model Metrics ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.metric("R² Score", f"{r_squared:.4f}") 
st.sidebar.metric("MAE (Avg Error)", "$11.02") 

st.sidebar.markdown("---")
st.sidebar.write("Algorithm: XGBoost Regressor")
st.sidebar.write(f"MLflow Model Version: Latest (Loaded from {MODEL_NAME})")
st.sidebar.markdown("---")
st.sidebar.write("Developed using MLflow + Streamlit")