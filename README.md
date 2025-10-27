# 📈 FAANG Stock Price Prediction using XGBoost and MLOps Deployment

## Project Overview
This project develops and deploys a highly accurate Machine Learning model to forecast the next-day closing prices of the five FAANG stocks (Meta, Apple, Amazon, Netflix, and Google). The solution utilizes a **robust XGBoost Regressor** trained on historical price data, volume, technical indicators (SMA, Beta), and fundamental financial metrics (P/E, EPS, Debt-to-Equity).

The entire pipeline—from data processing to model delivery—is orchestrated using **MLflow** for tracking and registration, culminating in a professional, real-time prediction engine deployed via **Streamlit** (featuring a Dark Theme UI).

### Key Results
* **Model:** XGBoost Regressor
* **Performance:** Achieved an outstanding **R² Score of 0.9842** on the test set, demonstrating excellent predictive capability.
* **Architecture:** End-to-End MLOps pipeline including artifact saving, model registration, and deployment.

---

## 🔬 Exploratory Data Analysis (EDA) & Feature Engineering

The EDA phase was crucial for transforming raw data into predictive features and understanding underlying stock market trends. (See the **`FAANG(2).ipynb`** notebook for detailed analysis and visualizations.)

### Key EDA Insights:
* **Time-Series Dependencies:** Confirmed a strong correlation between a stock's closing price and its lagged values (e.g., previous day's close), validating the creation of time-series features like `Close_Lag1`.
* **Technical Indicators:** Engineered features like the **10-day Simple Moving Average (SMA\_10)** and **Beta** provided key information about short-term momentum and market volatility.
* **Fundamental Influence:** Verified the predictive power of fundamental data such as **P/E Ratio**, **EPS (Earnings Per Share)**, and **Market Cap** in capturing long-term value and sentiment.
* **Feature Set:** The final training data included over 30 engineered features, including one-hot encodings for the specific stock tickers, allowing the model to learn company-specific patterns.

---

## 🧠 Machine Learning (ML) Pipeline

### Model Selection & Training
* **Algorithm:** **XGBoost Regressor** was selected for its efficiency, strong performance on structured data, and superior ability to handle complex feature interactions compared to linear models.
* **Strategy:** The model was trained to predict the **next-day closing price** using a comprehensive feature set including normalized price, volume, and fundamental data.
* **Evaluation:** After hyperparameter tuning, the model achieved the following metrics on the test set:
    * **Coefficient of Determination (R²):** **0.9842**
    * **Mean Absolute Error (MAE):** **$11.02** (representing the average error in USD)

### MLOps and Model Registration (MLflow)
Every step of the modeling process was tracked and managed using **MLflow**:
1.  **Artifact Saving:** The crucial `MinMaxScaler` object and the final feature column order (`feature_columns.csv`) were saved as essential deployment artifacts.
2.  **Model Registration:** The trained XGBoost model was registered in the MLflow Model Registry, versioned as **`FAANG_XGBoost_Price_Predictor/1`**.
3.  **Deployment Linkage:** The Streamlit application dynamically loads the scaler, feature list, and the model (by version number) from the registered artifacts, guaranteeing consistency between the training environment and the production application.

---

## 🛠️ Technical Stack & Tools

| Component | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Data Processing** | Pandas, NumPy | Data cleaning, feature engineering, and transformation. |
| **Feature Scaling** | Scikit-learn (MinMaxScaler) | Normalizing data for optimal model training. |
| **Modeling** | **XGBoost**, Scikit-learn | High-performance gradient boosting for prediction. |
| **MLOps & Tracking** | **MLflow** | Experiment tracking, model versioning, and artifact management. |
| **Deployment** | **Streamlit** | Interactive, real-time web application for predictions (Dark Theme UI). |

---

## 💻 How to Run the Application

This guide assumes you have cloned the repository and have Python and pip installed.

### Prerequisites

For the application to run correctly, your local project directory must contain these files, which are loaded directly by the Streamlit script:
1.  `Streamlit_app.py` (The main application code)
2.  `minmax_scaler.pkl` (The saved scaler object)
3.  `feature_columns.csv` (The order of features used during training)
4.  MLflow tracking folders (`mlruns/`) containing the registered model.

### Execution Steps

1.  **Install Dependencies:**
    ```bash
    pip install streamlit pandas numpy xgboost scikit-learn joblib mlflow
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd [YOUR_PROJECT_DIRECTORY]
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run Streamlit_app.py
    ```
The application will automatically open in your default browser at `http://localhost:8501`.
