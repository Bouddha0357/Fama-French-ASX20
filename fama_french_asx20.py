import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# === Step 1: Define ASX 20 tickers (Yahoo format ends with '.AX') ===
asx20 = [
    "BHP.AX", "CBA.AX", "WBC.AX", "NAB.AX", "ANZ.AX",
    "WOW.AX", "WES.AX", "TLS.AX", "CSL.AX", "MQG.AX",
    "FMG.AX", "TCL.AX", "BXB.AX", "GMG.AX", "STO.AX",
    "RIO.AX", "SUN.AX", "ORG.AX", "APA.AX", "ALL.AX"
]

# === Step 2: Download adjusted close prices ===
def get_prices(tickers, start="2020-01-01", end="2024-12-31"):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    if 'Adj Close' in data.columns:
        return data['Adj Close']  # flat structure
    elif isinstance(data.columns, pd.MultiIndex):
        return data.loc[:, pd.IndexSlice['Adj Close', :]].droplevel(0, axis=1)  # clean multi-index
    else:
        raise ValueError("Unexpected data structure returned by yfinance.")

# === Step 3: Calculate percentage returns ===
def calculate_returns(price_data):
    return price_data.pct_change().dropna()

# === Step 4: Load Fama-French factors ===
def load_fama_french(filepath):
    factors = pd.read_csv(filepath, index_col=0, parse_dates=True)
    factors = factors.loc[:, ['Mkt-RF', 'SMB', 'HML', 'RF']] / 100  # convert % to decimals
    return factors

# === Step 5: Run Fama-French regression for each stock ===
def fama_french_regression(stock_returns, factors):
    results = {}
    for ticker in stock_returns.columns:
        df = pd.concat([stock_returns[ticker], factors], axis=1).dropna()
        y = df[ticker] - df['RF']  # excess return
        X = df[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        results[ticker] = model
    return results

# === Step 6: Display regression results ===
def print_results(models):
    for ticker, model in models.items():
        print(f"\n=== Regression Results for {ticker} ===")
        print(model.summary())

# === MAIN EXECUTION ===
if __name__ == "__main__":
    prices = get_prices(asx20)
    returns = calculate_returns(prices)
    
    # Replace with path to your Fama-French AU factors file
    factors = load_fama_french("fama_french_au.csv")
    
    aligned_returns = returns.loc[returns.index.intersection(factors.index)]
    aligned_factors = factors.loc[aligned_returns.index]
    
    models = fama_french_regression(aligned_returns, aligned_factors)
    print_results(models)
