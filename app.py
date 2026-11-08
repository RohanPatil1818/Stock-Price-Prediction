
import streamlit as st
import pandas as pd
import os
import json
import plotly.graph_objects as go

from src.config import ASSETS
from src.agent.technical_levels import calculate_support_resistance, determine_position
from src.models.predictor import prepare_modeling_data, train_and_evaluate

st.set_page_config(page_title="AI Sentiment Arbitrage", layout="wide", initial_sidebar_state="expanded")

st.title("📈 AI Agent: Sentiment Arbitrage Dashboard")
st.markdown("Interactive visualization of $n+1$ price predictions based on multi-source Apify social & Exa.com news data.")

@st.cache_data
def load_and_process_data():
    results = []
    for exchange, assets in ASSETS.items():
        for asset in assets:
            ticker = asset["ticker"]
            file_path = f"data/processed/{ticker}_features.csv"
            raw_path = f"data/raw/{ticker}_raw.json"

            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                mentions = int(df["mentions_count"].iloc[-1]) if "mentions_count" in df.columns else 0
                if mentions == 0 and os.path.exists(raw_path):
                    with open(raw_path, "r", encoding="utf-8") as f:
                        raw_d = json.load(f)
                        mentions = len(raw_d.get("news", [])) + len(raw_d.get("social", []))

                if len(df) > 10:
                    latest_close = float(df["Close"].iloc[-1])
                    latest_sentiment = float(df["hourly_sentiment_score"].iloc[-1])

                    levels = calculate_support_resistance(df)
                    position = determine_position(latest_sentiment, latest_close, levels)

                    model_df = prepare_modeling_data(df)
                    clf, reg, feature_cols, cv_acc, cv_mae = train_and_evaluate(model_df, ticker)
                    latest_feat = df.iloc[-1:][feature_cols]
                    pred_dir = clf.predict(latest_feat)[0]
                    pred_ret = float(reg.predict(latest_feat)[0])
                    pred_price = latest_close * (1.0 + pred_ret)
                    direction = "UP 🟢" if pred_dir == 1 else "DOWN 🔴"

                    results.append({
                        "Exchange": exchange,
                        "Asset": asset["name"],
                        "Ticker": ticker,
                        "Current Price": round(latest_close, 2),
                        "Mentions": mentions,
                        "Sentiment": round(latest_sentiment, 4),
                        "Support (S1)": levels.get("support_1", 0.0),
                        "Resistance (R1)": levels.get("resistance_1", 0.0),
                        "Recommendation": position,
                        "Predicted n+1 Price": round(pred_price, 2),
                        "n+1 Direction": direction,
                        "OOS Accuracy": f"{cv_acc:.1%}",
                        "DataFrame": df
                    })
    return results

results = load_and_process_data()

if not results:
    st.warning("No processed data found. Run `python run_pipeline.py` first.")
else:
    st.markdown("### 📊 Market Overview")
    col1, col2, col3 = st.columns(3)
    bullish = len([r for r in results if "UP" in r["n+1 Direction"]])
    col1.metric("Total Assets Analyzed", len(results))
    col2.metric("Bullish Predictions (n+1)", bullish)
    col3.metric("Bearish Predictions (n+1)", len(results) - bullish)

    st.markdown("---")
    st.subheader("📋 Live Arbitrage Matrix (Sorted by Mentions & Sentiment per Exchange)")

    for exch in ["HKEX", "KRX"]:
        st.markdown(f"#### {exch} Exchange")
        exch_rows = [r for r in results if r["Exchange"] == exch]
        exch_rows.sort(key=lambda x: (x["Mentions"], x["Sentiment"]), reverse=True)
        summary_df = pd.DataFrame(exch_rows).drop(columns=["DataFrame"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Deep Dive: Price vs. Sentiment Overlay")

    selected_ticker = st.selectbox("Select Asset to Analyze:", [r["Ticker"] for r in results])
    selected_data = next(r for r in results if r["Ticker"] == selected_ticker)
    df_plot = selected_data["DataFrame"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Close"],
        name="Price (Local Cur.)",
        line=dict(color="#00b4d8", width=2)
    ))
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["hourly_sentiment_score"],
        name="Sentiment Score",
        yaxis="y2",
        marker_color="rgba(255, 159, 28, 0.55)"
    ))

    fig.update_layout(
        title=f"Time-Series Analysis: {selected_data['Asset']} ({selected_ticker})",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Engagement-Weighted Sentiment", overlaying="y", side="right", range=[-1.2, 1.2]),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)