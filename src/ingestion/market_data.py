
# Script to fetch the last 10days of pricing data 

import yfinance as yf
import pandas as pd
from src.config import ASSETS

def fetch_recent_data(ticker_symbol: str, days: int = 10) -> pd.DataFrame:
    """
    Fetches hourly price data for the last 'days' days.
    Hourly data is crucial for intraday sentiment alignment.
    """
    ticker = yf.Ticker(ticker_symbol)
    # 1h interval is supported for up to 730 days on yfinance
    df = ticker.history(period=f"{days}d", interval="1h")
    
    if df.empty:
        print(f"Warning: No data found for {ticker_symbol}")
        return df
        
    # Clean up timezone info to standard UTC for alignment with Apify/Exa
    df.index = pd.to_datetime(df.index, utc=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

if __name__ == "__main__":
    print("Testing HKEX Data Fetch (Tencent):")
    hk_data = fetch_recent_data("0700.HK", days=3)
    print(hk_data.tail(3))
    
    print("\nTesting KRX Data Fetch (Samsung):")
    kr_data = fetch_recent_data("005930.KS", days=3)
    print(kr_data.tail(3))