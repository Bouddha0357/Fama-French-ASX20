import requests
try:
    response = requests.get("https://finance.yahoo.com")
    if response.status_code == 200:
        print("Successfully connected to Yahoo Finance!")
    else:
        print(f"Connection failed with status code {response.status_code}")
except Exception as e:
    print(f"An error occurred while connecting: {e}")
