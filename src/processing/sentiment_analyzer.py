# src/processing/sentiment_analyzer.py
import math
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timezone

def get_basic_sentiment(text: str) -> float:
    """Returns a polarity score between -1.0 (negative) and 1.0 (positive)"""
    return TextBlob(str(text)).sentiment.polarity

def process_and_weight_sentiment(raw_data: dict) -> pd.DataFrame:
    """
    Takes raw JSON containing social and news arrays, calculates sentiment, 
    and weights it by engagement.
    """
    records = []
    
    # 1. Process Social Posts
    for post in raw_data.get("social", []):
        text = post.get("text", "")
        sentiment = get_basic_sentiment(text)
        
        # Clean engagement numbers (handle strings like "1.2K")
        try:
            views = int(str(post.get("views", "0")).replace("K", "000").replace(",", ""))
        except:
            views = 0
            
        # Logarithmic weighting prevents viral outliers
        weight = math.log1p(views) + 1.0 
        
        records.append({
            "timestamp_utc": post.get("date") or post.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
            "source": "social",
            "sentiment_raw": sentiment,
            "weight": weight,
            "weighted_sentiment": sentiment * weight
        })

    # 2. Process News 
    for article in raw_data.get("news", []):
        text = article.get("title", "") + " " + article.get("text", "")
        sentiment = get_basic_sentiment(text)
        weight = 5.0 # Flat baseline weight for institutional news
        
        records.append({
            "timestamp_utc": article.get("timestamp_utc") or article.get("published_date") or datetime.now(timezone.utc).isoformat(),
            "source": "news",
            "sentiment_raw": sentiment,
            "weight": weight,
            "weighted_sentiment": sentiment * weight
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        df = df.sort_values("timestamp_utc")
    return df