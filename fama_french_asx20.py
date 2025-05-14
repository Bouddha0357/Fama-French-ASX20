import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

st.set_page_config(page_title="Daily Fama-French Regression", layout="centered")
st.title("📈 Daily Fama-French 3-Factor Regression (ASX)")

# === File upload ===
uploaded_file = st.file_uploader("Upload Daily Fama-French Factor File (CSV)", type=["csv"])

# === Ticker input ===
ticker = st.text_input("Enter ASX Ticker (e.g., BHP.AX):", value="BHP.AX")

# === Date range ===
start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

if uploaded_file and ticker:
    try:
        # --- Load daily factors ---
        factors = pd.read_csv(uploaded_file)
        factors['Date'] = pd.to_datetime(factors['Date'])
        factors.set_index('Date', inplace=True)

        # Convert to decimal returns
        factors = factors[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100

        # --- Load daily stock data ---
        st.info(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

        if data.empty or 'Adj Close' not in data.columns:
            st.error("Stock data not found.")
        else:
            prices = data['Adj Close'].to_frame(name=ticker)
            daily_returns = prices.pct_change().dropna()

            # --- Align with factor dates ---
            aligned_returns = daily_returns.loc[daily_returns.index.intersection(factors.index)]
            aligned_factors = factors.loc[aligned_returns.index]

            # --- Regression ---
            df = pd.concat([aligned_returns[ticker], aligned_factors], axis=1).dropna()
            y = df[ticker] - df['RF']
            X = sm.add_constant(df[['Mkt-RF', 'SMB', 'HML']])
            model = sm.OLS(y, X).fit()

            st.subheader(f"Regression Results for {ticker}")
            st.text(model.summary())

    except Exception as e:
        st.error(f"An error occurred: {e}")
