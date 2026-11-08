
import pandas as pd
from rich.console import Console
from rich.table import Table
from src.agent.technical_levels import calculate_support_resistance, determine_position

def generate_terminal_report(ticker: str, df: pd.DataFrame):
    console = Console()
    
    # Initialize the visual table
    table = Table(
        title="HENG SENG & KRX SENTIMENT ARBITRAGE REPORT",
        show_header=True, 
        header_style="bold magenta"
    )
    
    table.add_column("Ticker", style="cyan", width=10)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Sentiment", justify="right", width=10)
    table.add_column("Position Recom.", style="yellow")
    table.add_column("S1 / R1", justify="center")

    # Extract latest metrics
    latest_close = df['Close'].iloc[-1]
    latest_sentiment = df['hourly_sentiment_score'].iloc[-1]
    
    # Calculate levels and position
    levels = calculate_support_resistance(df)
    position = determine_position(latest_sentiment, latest_close, levels)
    
    # Format S1/R1 string
    s1_r1 = f"{levels.get('support_1', 0.0)} / {levels.get('resistance_1', 0.0)}"

    # Add the row to our table
    table.add_row(
        ticker,
        f"{latest_close:.2f}",
        f"{latest_sentiment:.4f}",
        position,
        s1_r1
    )
    
    console.print(table)
    print("\n")