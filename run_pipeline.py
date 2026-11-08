
import os
import json
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

            try:
                news = exa.fetch_recent_news(name, days_back=10)
                social = apify.fetch_social_posts(
                    query=keyword,
                    ticker=ticker,
                    exchange=exchange,
                    days_back=10,
                    actor_id="apify/web-scraper"
                )
                raw_data = {"ticker": ticker, "exchange": exchange, "news": news, "social": social}
                with open(f"data/raw/{ticker}_raw.json", "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                console.print(f"[red]Error fetching {ticker}: {e}[/red]")
                continue

            sentiment_df = process_and_weight_sentiment(raw_data)
            master_df = align_features(ticker, sentiment_df, days=10)

            if master_df.empty:
                console.print(f"[yellow]Skipping {ticker} - insufficient market data.[/yellow]")
                continue

            master_df["mentions_count"] = len(news) + len(social)
            master_df.to_csv(f"data/processed/{ticker}_features.csv")

            latest_close = float(master_df["Close"].iloc[-1])
            latest_sentiment = float(master_df["hourly_sentiment_score"].iloc[-1])
            mentions = len(news) + len(social)

            levels = calculate_support_resistance(master_df)
            position = determine_position(latest_sentiment, latest_close, levels)

            model_df = prepare_modeling_data(master_df)
            if len(model_df) > 10:
                clf, reg, feature_cols, cv_acc, cv_mae = train_and_evaluate(model_df, ticker)
                latest_features = master_df.iloc[-1:][feature_cols]
                pred_dir = clf.predict(latest_features)[0]
                pred_ret = float(reg.predict(latest_features)[0])
                pred_price = latest_close * (1.0 + pred_ret)
                direction = "UP 🟢" if pred_dir == 1 else "DOWN 🔴"
            else:
                pred_price = latest_close
                direction = "N/A"
                cv_acc = 0.0

            results.append({
                "exchange": exchange,
                "ticker": ticker,
                "name": name,
                "price": latest_close,
                "mentions": mentions,
                "sentiment": latest_sentiment,
                "s1_r1": f"{levels.get('support_1', 0):.2f} / {levels.get('resistance_1', 0):.2f}",
                "position": position,
                "pred_price": pred_price,
                "direction": direction,
                "cv_acc": cv_acc
            })

    for exchange in ["HKEX", "KRX"]:
        exchange_data = [r for r in results if r["exchange"] == exchange]
        exchange_data.sort(key=lambda x: (x["mentions"], x["sentiment"]), reverse=True)

        table = Table(title=f"{exchange} Sentiment & Predictive Arbitrage Report", header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Name")
        table.add_column("Price", justify="right")
        table.add_column("Mentions", justify="right")
        table.add_column("Sentiment", justify="right")
        table.add_column("S1 / R1", justify="center")
        table.add_column("Position", style="yellow")
        table.add_column("n+1 Price", justify="right")
        table.add_column("n+1 Dir", justify="center")
        table.add_column("OOS Acc", justify="right")

        for r in exchange_data:
            table.add_row(
                r["ticker"],
                r["name"],
                f"{r['price']:.2f}",
                str(r["mentions"]),
                f"{r['sentiment']:+.4f}",
                r["s1_r1"],
                r["position"],
                f"{r['pred_price']:.2f}",
                r["direction"],
                f"{r['cv_acc']:.1%}"
            )
        console.print("\n")
        console.print(table)

if __name__ == "__main__":
    run_full_pipeline()