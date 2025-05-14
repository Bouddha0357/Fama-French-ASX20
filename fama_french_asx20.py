import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Define ASX 20 Tickers (adjusted for Yahoo Finance, e.g., "BHP.AX")
asx20 = ["BHP.AX", "CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX"]  # ...add all 20

# Step 2: Download stock prices
def get_prices(tickers, start="2020-01-01", end="2024-12-31"):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

# Step 3: Calculate log returns
def calculate_returns(price_data):
    return price_data.pct_change().dropna()

# Step 4: Load Fama-French Factors (AU version, as CSV)
def load_fama_french(filepath):
    factors = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return factors / 100  # Convert % to decimal

# Step 5: Run regression for each stock
def fama_french_regression(stock_returns, factors):
    results = {}
    for ticker in stock_returns.columns:
        df = pd.concat([stock_returns[ticker], factors], axis=1).dropna()
        y = df[ticker] - df['RF']  # Excess return
        X = df[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        results[ticker] = model
    return results

# Step 6: Output results
def print_results(models):
    for ticker, model in models.items():
        print(f"\nRegression Results for {ticker}:")
        print(model.summary())

# === MAIN ===
prices = get_prices(asx20)
returns = calculate_returns(prices)
factors = load_fama_french("fama_french_au.csv")
models = fama_french_regression(returns, factors)
print_results(models)
