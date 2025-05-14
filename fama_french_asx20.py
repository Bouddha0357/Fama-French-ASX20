import yfinance as yf

# Ticker symbol for Apple (AAPL)
ticker = "AAPL"

# Download stock data for Apple
data = yf.download(ticker, start="2022-01-01", end="2024-12-31", auto_adjust=True)

# Check if data is retrieved successfully
if data.empty or 'Adj Close' not in data.columns:
    print("Stock data not found.")
else:
    print(data.head())  # Display first few rows of stock data
