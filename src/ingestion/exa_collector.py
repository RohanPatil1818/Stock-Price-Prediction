# src/ingestion/exa_collector.py
import os
from datetime import datetime, timedelta, timezone
from exa_py import Exa
from dotenv import load_dotenv

load_dotenv()

class ExaNewsCollector:
    def __init__(self):
        key = os.getenv("EXA_API_KEY")
        if not key:
            raise ValueError("EXA_API_KEY is missing in .env")
        self.exa = Exa(key)

    def fetch_recent_news(self, asset_name: str, days_back: int = 10):
        print(f"[*] Fetching Exa news for: {asset_name}")
        
        # Exa requires YYYY-MM-DD format for date filtering
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Neural search query tailored for financial context
        query = f"{asset_name} stock market news OR earnings"
        
        try:
            # Relaxed parameters to guarantee a data payload
            response = self.exa.search_and_contents(
                query=f"{asset_name} stock market news",
                type="keyword",       # Neural search understands context, not just exact keywords
                num_results=3,       # Top 3 most relevant articles
                text=True            # We need the actual article text for the LLM later
            )
            
            news_items = []
            for result in response.results:
                news_items.append({
                    "title": result.title,
                    "url": result.url,
                    "timestamp_utc": result.published_date,
                    "text": result.text[:1500] if result.text else "", # Cap at 1500 chars to save LLM tokens
                    "source": "exa_news"
                })
            return news_items
        except Exception as e:
            print(f"[!] Error fetching Exa data for {asset_name}: {e}")
            return []