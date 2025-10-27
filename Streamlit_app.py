import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow

# --- Configuration and Artifact Loading ---
try:
    # 1. Load the Scaler
    scaler = joblib.load('minmax_scaler.pkl')

    # 2. Load the Feature Columns
    feature_columns = pd.read_csv('feature_columns.csv')['FeatureName'].tolist()

    # 3. Load the Model from MLflow Registry 
    MODEL_NAME = "FAANG_XGBoost_Price_Predictor"
    model_uri = f"models:/{MODEL_NAME}/1"
    model = mlflow.sklearn.load_model(model_uri)
    LOAD_SUCCESS = True
except Exception as e:
    LOAD_SUCCESS = False
    MODEL_LOAD_ERROR = f"FATAL ERROR: Could not load model artifacts. Ensure all files (.pkl, .csv) are present and MLflow is accessible. Details: {e}"


# --- Streamlit App Configuration (st.set_page_config MUST be the FIRST Streamlit command) ---
st.set_page_config(page_title="FAANG Stock Predictor", layout="wide")

# 2. Apply Custom Styling (Dark Theme CSS)
st.markdown("""
<style>
/* 1. Global Dark Background and Light Text */
.stApp {
    background-color: #1a1a2e; /* Dark Blue/Purple Background */
    color: #f0f2f6; /* Light Gray Text */
}
/* Ensure main content is light text */
.main {
    color: #f0f2f6; 
}
/* 2. Style Headings (White or Accent Color) */
h1, h2, h3 {
    color: #00ffaa; /* Bright Mint Green for headings */
}
/* 3. Style Metrics (Must contrast with dark background) */
.stMetric {
    background-color: #272744; /* Slightly lighter dark background for contrast */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); /* Darker shadow */
    border-left: 5px solid #00ffaa; /* Bright green highlight */
    color: #f0f2f6; /* Light text inside metric boxes */
}
.stMetric > div > div > div:first-child {
    color: #a0a0ff; /* Light blue/purple for metric label */
}
.stMetric > div > div > div:nth-child(2) {
    color: #00ffaa; /* Bright green for metric value */
}
</style>
""", unsafe_allow_html=True)


# --- Application Title and Error Check ---
st.title("📈 FAANG Next-Day Closing Price Prediction Engine")

if not LOAD_SUCCESS:
    st.error(MODEL_LOAD_ERROR)
    st.stop()


# --- Global Stock Ticker Definition ---
# Define stock_ticker before the main logic starts, giving it a default value
stock_ticker = None


# --- Main Application Layout ---
col_main, col_sidebar = st.columns([3, 1])

with col_main:
    st.header("Prediction Parameters")
    st.markdown("Adjust the key financial and technical indicators to generate a forecast.")

    # --- Input Section organized with Columns ---
    
    # 1. Ticker Selection is defined first
    stock_ticker = st.selectbox("Select Stock Ticker:", ('AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX'))

    st.subheader("Time-Series Inputs (High Predictive Power)")
    col1, col2 = st.columns(2)
    
    with col1:
        close_lag1 = st.number_input(
            "1. Previous Day's Closing Price ($)", 
            value=150.00, 
            min_value=0.01, 
            format="%.2f",
            help="The stock's closing price on the last trading day."
        )
    
    with col2:
        sma_10 = st.number_input(
            "2. 10-Day Simple Moving Average ($)", 
            value=148.50, 
            min_value=0.01, 
            format="%.2f",
            help="The average closing price over the last 10 days."
        )

    st.subheader("Fundamental & Volume Inputs")
    col3, col4, col5 = st.columns(3)

    with col3:
        volume = st.number_input("Volume (Last Day)", value=100_000_000, min_value=0, step=100000)
        pe_ratio = st.number_input("P/E Ratio", value=30.0, min_value=0.1, format="%.2f")
        roe = st.slider("Return on Equity (ROE)", min_value=-0.5, max_value=0.5, value=0.2, step=0.01)

    with col4:
        market_cap = st.number_input("Market Cap (in billions)", value=2_500_000_000_000, min_value=1_000_000_000)
        beta = st.slider("Beta", min_value=0.0, max_value=3.0, value=1.2, step=0.05)
        debt_to_equity = st.slider("Debt to Equity Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    with col5:
        eps = st.number_input("EPS (Earnings Per Share)", value=6.50, format="%.2f")
        total_debt = st.number_input("Total Debt (Approximate)", value=100_000_000_000, min_value=0)

    st.markdown("---")
    
    # --- Prediction Button and Logic ---
    if st.button("Generate Price Forecast", key='predict_button', type='primary'):
        
        with st.spinner('Calculating XGBoost forecast...'):
            
            # 1. Create a dictionary to hold ALL 31 input features (unscaled)
            input_data = {}
            for col in feature_columns:
                input_data[col] = 0.0 

            # Fill in user-specified numerical values
            input_data['Volume'] = volume
            input_data['Market Cap'] = market_cap
            input_data['PE Ratio'] = pe_ratio
            input_data['Beta'] = beta
            input_data['EPS'] = eps
            input_data['Total Debt'] = total_debt
            input_data['Return on Equity (ROE)'] = roe
            input_data['Debt to Equity'] = debt_to_equity
            input_data['Close_Lag1'] = close_lag1
            input_data['SMA_10'] = sma_10
            # Placeholder values for remaining features 
            input_data['Forward PE'] = 0.0
            input_data['Net Income'] = 0
            input_data['Current Ratio'] = 0.0
            input_data['Dividends Paid'] = 0.0
            input_data['Dividend Yield'] = 0.0
            input_data['Quarterly Revenue Growth'] = 0.0
            input_data['Target Price'] = 0.0
            input_data['Free Cash Flow'] = 0
            input_data['Operating Margin'] = 0.0
            input_data['Profit Margin'] = 0.0
            input_data['Quick Ratio'] = 0.0
            input_data['Price to Book Ratio'] = 0.0
            input_data['Enterprise Value'] = 0
            input_data['Beta (5Y)'] = 0.0
            input_data['Annual Dividend Rate'] = 0.0
            
            # Set the one-hot encoded columns based on the selection
            input_data[f'Stock_{stock_ticker}'] = 1.0
            input_data['Analyst_buy'] = 1.0 

            # 2. Convert to DataFrame, Reorder, and Scale
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_columns] 
            
            numerical_cols_to_scale = input_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_exclude = [col for col in input_df.columns if col.startswith('Stock_') or col.startswith('Analyst_')]
            numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col not in cols_to_exclude]

            input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])

            # 3. Make the Prediction
            prediction = model.predict(input_df)[0]
            
            # 4. Display Result
            st.subheader(f"Forecast Result for {stock_ticker}")
            
            prediction_delta = prediction - close_lag1
            
            st.metric(
                label=f"Predicted Closing Price for {stock_ticker} tomorrow:",
                value=f"${prediction:,.2f}",
                delta=f"{prediction_delta:,.2f} USD change"
            )
            
            st.success("Forecast generated successfully!")

# --- Sidebar for Model Info ---
with col_sidebar:
    st.subheader("Model Metadata")
    
    # Check if stock_ticker was successfully defined by the selectbox
    if stock_ticker:
        st.info(f"**Selected Stock:** {stock_ticker}")
    else:
        st.warning("Select a stock to view model details.")

    st.markdown("---")
    
    # Display key model evaluation metrics
    st.markdown("**Model Performance Metrics**")
    st.metric("Model R² Score", value="0.9842")
    st.metric("MAE (Avg Error)", value="$11.02")

    st.markdown("---")
    st.markdown("**System Details**")
    st.markdown(f"**Algorithm:** XGBoost Regressor")
    st.markdown(f"**MLflow Version:** 1")
    st.markdown(f"**Data Range:** 2005 - Present")