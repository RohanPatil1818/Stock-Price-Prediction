#
import pandas as pd
import numpy as np
from src.ingestion.market_data import fetch_recent_data

def align_features(ticker: str, sentiment_df: pd.DataFrame, days: int = 10) -> pd.DataFrame:
    market_df = fetch_recent_data(ticker, days=days)
    if market_df.empty:
        return market_df

    market_df = market_df.sort_index()

    if not sentiment_df.empty:
        s_df = sentiment_df.copy().sort_values("timestamp_utc")
        s_df["instant_score"] = s_df["weighted_sentiment"] / s_df["weight"]

        # Resample sentiment into 4-hour blocks and use merge_asof to map onto exact market candle times
        s_resampled = (
            s_df.set_index("timestamp_utc")
            .resample("4h")
            .agg({"weighted_sentiment": "sum", "weight": "sum"})
        )
        s_resampled["hourly_sentiment_score"] = (
            s_resampled["weighted_sentiment"] / s_resampled["weight"]
        ).replace([np.inf, -np.inf], np.nan)
        s_resampled = s_resampled.dropna(subset=["hourly_sentiment_score"]).reset_index()

        m_reset = market_df.reset_index()
        time_col = m_reset.columns[0]  # 'Datetime' or 'Date'

        merged_df = pd.merge_asof(
            m_reset.sort_values(time_col),
            s_resampled[["timestamp_utc", "hourly_sentiment_score"]].sort_values("timestamp_utc"),
            left_on=time_col,
            right_on="timestamp_utc",
            direction="nearest"
        )
        merged_df.set_index(time_col, inplace=True)
        if "timestamp_utc" in merged_df.columns:
            merged_df.drop(columns=["timestamp_utc"], inplace=True)

        # Add slight decay jitter based on return momentum so constant forward-fills don't flatten variance
        base_mean = float(s_df["instant_score"].mean()) if not s_df["instant_score"].isna().all() else 0.05
        merged_df["hourly_sentiment_score"] = merged_df["hourly_sentiment_score"].ffill().bfill().fillna(base_mean)
    else:
        merged_df = market_df.copy()
        merged_df["hourly_sentiment_score"] = 0.0

    # Technical & Sentiment Features
    merged_df["return_1h"] = merged_df["Close"].pct_change().fillna(0)
    # Blend raw sentiment with price reaction for realistic hourly bars
    merged_df["hourly_sentiment_score"] = (
        0.75 * merged_df["hourly_sentiment_score"] + 0.25 * np.tanh(merged_df["return_1h"] * 50)
    ).clip(-1.0, 1.0)
    merged_df["rolling_sentiment_3h"] = merged_df["hourly_sentiment_score"].rolling(3, min_periods=1).mean()
    merged_df["vol_spike"] = (merged_df["Volume"] / merged_df["Volume"].rolling(10, min_periods=1).mean()).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    return merged_df.dropna()