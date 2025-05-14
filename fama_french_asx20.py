import streamlit as st
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

st.set_page_config(page_title="Fama-French Regression", layout="centered")

st.title("📊 Fama-French 3-Factor Regression (ASX)")

# === 1. File upload ===
uploaded_file = st.file_uploader("Upload your Fama-French CSV (must include 'Mkt-RF', 'SMB', 'HML', 'RF')", type=["csv"])

# === 2. Ticker input ===
ticker = st.text_input("Enter ASX Ticker (e.g., BHP.AX):", value="BHP.AX")

# === 3. Date selection ===
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# === 4. Process when file and ticker are ready ===
if uploaded_file and ticker:
    try:
        # Load factor data
        factors = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        required_cols = {"Mkt-RF", "SMB", "HML", "RF"}
        if not required_cols.issubset(factors.columns):
            st.error(f"CSV file must include columns: {required_cols}")
        else:
            factors = factors[list(required_cols)] / 100  # Convert % to decimals

            # Load stock price
            st.info(f"Downloading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if data.empty or 'Adj Close' not in data.columns:
                st.error("Failed to download valid price data. Check ticker symbol or internet connection.")
            else:
                prices = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                returns = prices.pct_change().dropna()

                # Align with factors
                aligned_returns = returns.loc[returns.index.intersection(factors.index)]
                aligned_factors = factors.loc[aligned_returns.index]

                if aligned_returns.empty:
                    st.error("No overlapping dates between price data and factors.")
                else:
                    # Regression
                    df = pd.concat([aligned_returns[ticker], aligned_factors], axis=1).dropna()
                    y = df[ticker] - df['RF']
                    X = df[['Mkt-RF', 'SMB', 'HML']]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()

                    # Output
                    st.subheader(f"Regression Results for {ticker}")
                    st.text(model.summary())
    except Exception as e:
        st.error(f"An error occurred: {e}")
