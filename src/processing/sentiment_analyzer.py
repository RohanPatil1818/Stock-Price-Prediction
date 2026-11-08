
import math
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta, timezone

BULLISH_TERMS = ["surge", "jump", "beat", "growth", "buy", "upgrade", "profit", "bull", "gain", "rally", "strong", "record", "outperform", "rise", "상승", "매수", "호재"]
BEARISH_TERMS = ["drop", "fall", "miss", "loss", "sell", "downgrade", "crash", "bear", "decline", "weak", "cut", "lawsuit", "risk", "slump", "하락", "매도", "악재"]

def get_financial_sentiment(text: str) -> float:
    """Combines TextBlob polarity with financial domain keyword scoring."""
    clean_text = str(text).lower()
    base_polarity = TextBlob(str(text)).sentiment.polarity

    bull_hits = sum(1 for word in BULLISH_TERMS if word in clean_text)
    bear_hits = sum(1 for word in BEARISH_TERMS if word in clean_text)
    lexicon_score = (bull_hits - bear_hits) * 0.18

    combined = base_polarity + lexicon_score
    return max(-1.0, min(1.0, combined))

def _parse_int(val, default=0) -> int:
    try:
        s = str(val).upper().replace(",", "").strip()
        if "K" in s:
            return int(float(s.replace("K", "")) * 1000)
        if "M" in s:
            return int(float(s.replace("M", "")) * 1000000)
        return int(float(s))
    except Exception:
        return default

def process_and_weight_sentiment(raw_data: dict) -> pd.DataFrame:
    """
    Calculates sentiment across Social (2 sources) and Exa News,
    weighting by likes, reposts, views, and author_followers.
    """
    records = []
    now_utc = datetime.now(timezone.utc)

    # 1. Process Social Posts
    for idx, post in enumerate(raw_data.get("social", [])):
        text = post.get("text", "")
        sentiment = get_financial_sentiment(text)

        views = _parse_int(post.get("views", 50), 50)
        likes = _parse_int(post.get("likes", 5), 5)
        reposts = _parse_int(post.get("reposts", 1), 1)
        followers = _parse_int(post.get("author_followers", 200), 200)

        # Engagement + Author Follower weighting
        weight = math.log1p(likes + 2 * reposts + 0.05 * views + 0.01 * followers) + 1.0

        raw_ts = post.get("timestamp_utc") or post.get("date")
        if not raw_ts:
            raw_ts = (now_utc - timedelta(hours=idx * 5)).isoformat()

        records.append({
            "timestamp_utc": raw_ts,
            "source": post.get("source", "social"),
            "sentiment_raw": sentiment,
            "weight": weight,
            "weighted_sentiment": sentiment * weight
        })

    # 2. Process Exa News
    for idx, article in enumerate(raw_data.get("news", [])):
        text = f"{article.get('title', '')} {article.get('text', '')}"
        sentiment = get_financial_sentiment(text)
        weight = 5.0  # Institutional news weight

        raw_ts = article.get("timestamp_utc") or article.get("published_date")
        # If Exa keyword search omits timestamp, stagger across recent trading days
        if not raw_ts:
            raw_ts = (now_utc - timedelta(hours=(idx + 1) * 12)).isoformat()

        records.append({
            "timestamp_utc": raw_ts,
            "source": "exa_news",
            "sentiment_raw": sentiment,
            "weight": weight,
            "weighted_sentiment": sentiment * weight
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["timestamp_utc"] = df["timestamp_utc"].fillna(now_utc)
        df = df.sort_values("timestamp_utc")
    return df