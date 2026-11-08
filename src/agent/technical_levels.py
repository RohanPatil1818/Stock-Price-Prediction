# src/agent/technical_levels.py
import pandas as pd

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Computes Standard Pivot Points and Support/Resistance levels based on recent price action.
    """
    # Ensure we have enough data, otherwise take what we have
    actual_window = min(window, len(df))
    if actual_window == 0:
        return {}

    recent_high = df['High'].tail(actual_window).max()
    recent_low = df['Low'].tail(actual_window).min()
    current_close = df['Close'].iloc[-1]
    
    # Standard Pivot Point math
    pivot = (recent_high + recent_low + current_close) / 3
    r1 = (2 * pivot) - recent_low
    s1 = (2 * pivot) - recent_high
    r2 = pivot + (recent_high - recent_low)
    s2 = pivot - (recent_high - recent_low)
    
    return {
        "current_price": round(float(current_close), 2),
        "pivot": round(float(pivot), 2),
        "support_1": round(float(s1), 2),
        "support_2": round(float(s2), 2),
        "resistance_1": round(float(r1), 2),
        "resistance_2": round(float(r2), 2)
    }

def determine_position(sentiment_score: float, current_price: float, sr_levels: dict) -> str:
    """
    Agent Logic: Combines sentiment direction and technical levels to advise a position.
    """
    if not sr_levels:
        return "Hold / Neutral (Insufficient Data)"

    s1 = sr_levels["support_1"]
    r1 = sr_levels["resistance_1"]
    
    # Bullish sentiment scenarios
    if sentiment_score > 0.15:
        if current_price < s1 * 1.02: # Price is near or below support 1
            return "Strong Buy / Long (Value Bounce)"
        elif current_price > r1:
            return "Tactical Long (Momentum Breakout)"
        else:
            return "Long (Accumulate)"
            
    # Bearish sentiment scenarios
    elif sentiment_score < -0.15:
        if current_price > r1 * 0.98: # Price is near or above resistance 1
            return "Strong Short (Resistance Rejection)"
        else:
            return "Underweight / Reduce Exposure"
            
    # Neutral zone
    else:
        return "Hold / Neutral"