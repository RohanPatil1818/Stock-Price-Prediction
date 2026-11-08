# run_pipeline.py
import os
import json
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import ASSETS
from src.ingestion.apify_collector import ApifySocialCollector
from src.ingestion.exa_collector import ExaNewsCollector
from src.processing.sentiment_analyzer import process_and_weight_sentiment
from src.processing.technical_features import align_features
from src.agent.technical_levels import calculate_support_resistance, determine_position
from src.models.predictor import prepare_modeling_data, train_and_evaluate

def run_full_pipeline():
    apify = ApifySocialCollector()
    exa = ExaNewsCollector()
    console = Console()
    
    results = []
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    for exchange, assets in ASSETS.items():
        console.print(f"\n[bold cyan]--- Processing {exchange} Assets ---[/bold cyan]")
        
        for asset in assets:
            ticker = asset["ticker"]
            name = asset["name"]
            keyword = asset["keywords"][0]
            console.print(f"[*] Extracting data for {name} ({ticker})...")
            
            # 1. Ingestion
            try:
                news = exa.fetch_recent_news(name, days_back=10)
                # Note: Using generic scraper for the loop; in production ensure actor targets Telegram/Naver
                social = apify.fetch_social_posts(query=keyword, days_back=10)
                
                raw_data = {"ticker": ticker, "news": news, "social": social}
                with open(f"data/raw/{ticker}_raw.json", "w") as f:
                    json.dump(raw_data, f)
            except Exception as e:
                console.print(f"[red]Error fetching {ticker}: {e}[/red]")
                continue

            # 2. Feature Engineering
            sentiment_df = process_and_weight_sentiment(raw_data)
            master_df = align_features(ticker, sentiment_df, days=10)
            
            if master_df.empty:
                console.print(f"[yellow]Skipping {ticker} - insufficient market data.[/yellow]")
                continue
                
            master_df.to_csv(f"data/processed/{ticker}_features.csv")
            
            # 3. Agent Technicals
            latest_close = master_df['Close'].iloc[-1]
            latest_sentiment = master_df['hourly_sentiment_score'].iloc[-1]
            mentions = len(news) + len(social)
            
            levels = calculate_support_resistance(master_df)
            position = determine_position(latest_sentiment, latest_close, levels)
            
            # 4. ML Prediction (n+1)
            model_df = prepare_modeling_data(master_df)
            if len(model_df) > 10:
                model, feature_cols = train_and_evaluate(model_df, ticker)
                latest_features = master_df.iloc[-1:][feature_cols]
                pred = model.predict(latest_features)[0]
                direction = "UP 🟢" if pred == 1 else "DOWN 🔴"
            else:
                direction = "N/A"

            results.append({
                "exchange": exchange,
                "ticker": ticker,
                "name": name,
                "price": latest_close,
                "mentions": mentions,
                "sentiment": latest_sentiment,
                "position": position,
                "direction": direction
            })

    # 5. Generate Sorted Visual Dashboard
    for exchange in ["HKEX", "KRX"]:
        exchange_data = [r for r in results if r["exchange"] == exchange]
        # Sort by Sentiment Score descending
        exchange_data.sort(key=lambda x: x["sentiment"], reverse=True)
        
        table = Table(title=f"{exchange} Sentiment & Predictive Arbitrage Report", header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Price", justify="right")
        table.add_column("Mentions", justify="right")
        table.add_column("Sentiment", justify="right")
        table.add_column("Agent Position", style="yellow")
        table.add_column("n+1 Predict", justify="center")
        
        for r in exchange_data:
            table.add_row(
                r["ticker"],
                f"{r['price']:.2f}",
                str(r["mentions"]),
                f"{r['sentiment']:.4f}",
                r["position"],
                r["direction"]
            )
        console.print("\n")
        console.print(table)

if __name__ == "__main__":
    run_full_pipeline()