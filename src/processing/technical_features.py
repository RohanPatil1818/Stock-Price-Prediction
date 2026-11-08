
import pandas as pd
from src.ingestion.market_data import fetch_recent_data

def align_features(ticker: str, sentiment_df: pd.DataFrame, days: int = 10) -> pd.DataFrame:
    # 1. Fetch Market Data (Hourly)
    market_df = fetch_recent_data(ticker, days=days)
    if market_df.empty:
        return market_df

    # 2. Aggregate Sentiment to Hourly Bars
    if not sentiment_df.empty:
        # Set time as index and resample to 1 Hour (1h)
        sentiment_df.set_index('timestamp_utc', inplace=True)
        
        hourly_sentiment = sentiment_df.resample('1h').agg({
            'weighted_sentiment': 'sum',
            'weight': 'sum'
        })
        # Calculate the true weighted average for the hour
        hourly_sentiment['hourly_sentiment_score'] = (
            hourly_sentiment['weighted_sentiment'] / hourly_sentiment['weight']
        ).fillna(0)
    else:
        hourly_sentiment = pd.DataFrame(columns=['hourly_sentiment_score'])

    # 3. Merge Market Data with Sentiment Data
    # Market data index is already timezone-aware UTC from Day 1
    merged_df = market_df.join(hourly_sentiment[['hourly_sentiment_score']], how='left')
    
    # Forward-fill sentiment for hours with price action but no news/posts, then fill remaining with 0
    merged_df['hourly_sentiment_score'] = merged_df['hourly_sentiment_score'].ffill().fillna(0)
    
    # 4. Add Technical Indicators for the ML Model
    merged_df['return_1h'] = merged_df['Close'].pct_change()
    merged_df['rolling_sentiment_3h'] = merged_df['hourly_sentiment_score'].rolling(3).mean()
    merged_df['vol_spike'] = merged_df['Volume'] / merged_df['Volume'].rolling(10).mean()
    
    return merged_df.dropna()