# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# Import our existing backend logic
from src.config import ASSETS
from src.agent.technical_levels import calculate_support_resistance, determine_position
from src.models.predictor import prepare_modeling_data, train_and_evaluate

# Configure the web page
st.set_page_config(page_title="AI Sentiment Arbitrage", layout="wide", initial_sidebar_state="expanded")

st.title("📈 AI Agent: Sentiment Arbitrage Dashboard")
st.markdown("Interactive visualization of $n+1$ price predictions based on alternative social & news data.")

@st.cache_data # Caches the data so the app runs lightning fast
def load_and_process_data():
    results = []
    for exchange, assets in ASSETS.items():
        for asset in assets:
            ticker = asset["ticker"]
            file_path = f"data/processed/{ticker}_features.csv"
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if len(df) > 10:
                    latest_close = df['Close'].iloc[-1]
                    latest_sentiment = df['hourly_sentiment_score'].iloc[-1]
                    
                    levels = calculate_support_resistance(df)
                    position = determine_position(latest_sentiment, latest_close, levels)
                    
                    # Run ML prediction on the fly
                    model_df = prepare_modeling_data(df)
                    model, feature_cols = train_and_evaluate(model_df, ticker)
                    pred = model.predict(df.iloc[-1:][feature_cols])[0]
                    direction = "UP 🟢" if pred == 1 else "DOWN 🔴"
                    
                    results.append({
                        "Exchange": exchange,
                        "Asset": asset["name"],
                        "Ticker": ticker,
                        "Price": round(latest_close, 2),
                        "Sentiment": round(latest_sentiment, 4),
                        "Recommendation": position,
                        "n+1 Predict": direction,
                        "DataFrame": df # Store for plotting
                    })
    return results

results = load_and_process_data()

if not results:
    st.warning("No processed data found. Run `python run_pipeline.py` first.")
else:
    # 1. Top Level Metrics
    st.markdown("### 📊 Market Overview")
    col1, col2, col3 = st.columns(3)
    bullish = len([r for r in results if "UP" in r["n+1 Predict"]])
    col1.metric("Total Assets Analyzed", len(results))
    col2.metric("Bullish Predictions (n+1)", bullish)
    col3.metric("Bearish Predictions (n+1)", len(results) - bullish)

    st.markdown("---")

    # 2. Interactive Data Table
    st.subheader("📋 Live Arbitrage Matrix")
    summary_df = pd.DataFrame(results).drop(columns=["DataFrame"])
    
    # Style the dataframe in Streamlit
    st.dataframe(
        summary_df, 
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # 3. Interactive Charting
    st.subheader("🔍 Deep Dive: Price vs. Sentiment Overlay")
    
    # Dropdown selector
    selected_ticker = st.selectbox("Select Asset to Analyze:", [r["Ticker"] for r in results])
    selected_data = next(r for r in results if r["Ticker"] == selected_ticker)
    df_plot = selected_data["DataFrame"]

    # Plotly Dual-Axis Chart
    fig = go.Figure()
    
    # Left Axis: Price
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['Close'], 
        name='Price (Local Cur.)', 
        line=dict(color='#00b4d8', width=2)
    ))
    
    # Right Axis: Sentiment Score (Displayed as bars)
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot['hourly_sentiment_score'], 
        name='Sentiment Score', 
        yaxis='y2', 
        marker_color='rgba(255, 159, 28, 0.5)'
    ))

    fig.update_layout(
        title=f"Time-Series Analysis: {selected_data['Asset']} ({selected_ticker})",
        yaxis=dict(title='Price'),
        yaxis2=dict(title='Engagement-Weighted Sentiment', overlaying='y', side='right', range=[-1.5, 1.5]),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)