
import os
from datetime import datetime, timedelta, timezone
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

class ApifySocialCollector:
    def __init__(self):
        # Initializes the client using the token from your .env file
        token = os.getenv("APIFY_API_TOKEN")
        if not token:
            raise ValueError("APIFY_API_TOKEN is missing in .env")
        self.client = ApifyClient(token)

    def fetch_social_posts(self, query: str, days_back: int = 10, actor_id: str = "apify/instagram-scraper"):
        """
        Note: For the assessment, you might use 'apify/web-scraper' for forums like LIHKG/Naver, 
        or a specific Telegram scraper. We default to a generic search structure here.
        """
        print(f"[*] Fetching social data for: {query}")
        
        # 1. Strict UTC Time Windowing
        since_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    
        
        # 2. Actor payload configuration - Now with a pageFunction!
        run_input = {
            "startUrls": [{"url": "https://t.me/s/hkstocks"}], 
            "maxRequestsPerCrawl": 1,
            "pageFunction": """
                async function pageFunction(context) {
                    const posts = [];
                    // Find all Telegram messages on the page
                    document.querySelectorAll('.tgme_widget_message').forEach((el) => {
                        const text = el.querySelector('.tgme_widget_message_text')?.innerText || '';
                        const date = el.querySelector('.tgme_widget_message_date time')?.getAttribute('datetime') || new Date().toISOString();
                        const views = el.querySelector('.tgme_widget_message_views')?.innerText || '0';
                        
                        if (text.length > 5) {
                            posts.push({ 
                                text: text, 
                                date: date, 
                                views: views,
                                platform: 'telegram'
                            });
                        }
                    });
                    return posts;
                }
            """
        }
        
        try:
            # 3. Trigger the scrape job
            run = self.client.actor(actor_id).call(run_input=run_input)
            
            # Safely extract dataset ID whether the SDK returned a dict or an object
            if isinstance(run, dict):
                dataset_id = run.get("defaultDatasetId")
            else:
                dataset_id = getattr(run, "defaultDatasetId", getattr(run, "default_dataset_id", None))
                
            dataset_items = self.client.dataset(dataset_id).list_items().items
            
            # If our pageFunction returned an array of posts, they might be nested. 
            # Let's flatten them just in case.
            # Flatten the nested results from Apify's web-scraper output
            flat_items = []
            for item in dataset_items:
                # Web-scraper puts our JS return value in 'pageFunctionResult'
                if isinstance(item, dict) and "pageFunctionResult" in item:
                    result = item["pageFunctionResult"]
                    if isinstance(result, list):
                        flat_items.extend(result)
                    elif result:
                        flat_items.append(result)
                elif isinstance(item, list):
                    flat_items.extend(item)
                else:
                    flat_items.append(item)
            
            # 4. Data Normalization
            normalized = []
            for item in dataset_items:
                # Extract date and force it to be recognizable
                raw_date = item.get("date") or item.get("createdAt") or datetime.now(timezone.utc).isoformat()
                
                normalized.append({
                    "timestamp_utc": raw_date,
                    "text": item.get("text") or item.get("caption") or item.get("content", ""),
                    "likes": item.get("likesCount") or item.get("likes", 0),
                    "reposts": item.get("sharesCount") or item.get("reposts", 0),
                    "views": item.get("viewCount") or item.get("views", 0),
                    "source": "apify_social"
                })
            return normalized
        except Exception as e:
            print(f"[!] Error fetching Apify data for {query}: {e}")
            return []