import yfinance as yf

try:
    # Download stock data for AAPL
    print("Fetching data for AAPL...")
    data = yf.download("AAPL", start="2022-01-01", end="2024-12-31", auto_adjust=True)
    
    if data.empty or 'Adj Close' not in data.columns:
        print("Stock data not found.")
    else:
        print("Stock data fetched successfully.")
        print(data.head())  # Print the first few rows of the stock data

except Exception as e:
    print(f"An error occurred: {e}")
