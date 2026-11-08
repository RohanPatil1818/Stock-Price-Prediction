# HengSeng Sentiment Price Prediction Agent

An end-to-end Python pipeline that ingests alternative social data and institutional news to predict $n+1$ price movements for equities on the HKEX and KRX exchanges.

## Architecture
1. **Ingestion Layer:** Apify (Social forums/Telegram) and Exa.com (News). Data is timestamp-aligned to strict UTC.
2. **NLP & Feature Engineering:** Implements an Engagement-Weighted Sentiment Score to prioritize high-visibility posts over zero-view spam. Aligned to Yahoo Finance hourly bars.
3. **Agent Node:** Calculates Standard Pivot Points and combines them with sentiment momentum to output tactical position recommendations (Long/Short/Neutral).
4. **Predictive Model:** LightGBM classifier predicting $n+1$ direction, validated using rigorous `TimeSeriesSplit` to prevent lookahead bias.

## Setup & Execution
1. Clone the repository and install requirements: `pip install -r requirements.txt`
2. Add API keys to `.env`: