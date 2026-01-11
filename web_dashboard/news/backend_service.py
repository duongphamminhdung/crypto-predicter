import time
import json
import threading
from datetime import datetime

# Import modules
from main import analyze_batch_with_retry
from fetch_rapidapi_x import fetch_real_tweets
from fetch_real_news import fetch_crypto_news

class CryptoSentimentService:
    def __init__(self):
        self.LIMIT_TWEETS = 10
        self.LIMIT_NEWS = 5
        self.BATCH_SIZE = 10
        self.is_running = False

    def fetch_data(self, sources):
        all_items = []
        print(f"\n--- 📥 Collecting RICH Data from: {', '.join(sources)} ---")

        # 1. Fetch Twitter (Now returns Rich Dicts)
        if "twitter" in sources:
            try:
                # Calls the NEW fetch_rapidapi_x
                rich_tweets = fetch_real_tweets(query="Bitcoin", limit=self.LIMIT_TWEETS)
                all_items.extend(rich_tweets)
            except Exception as e:
                print(f"   ❌ Twitter Fetch Failed: {e}")

        # 2. Fetch News (Now returns Rich Dicts with Favicon)
        if "news" in sources:
            try:
                # Calls the NEW fetch_real_news
                rich_news = fetch_crypto_news(limit=self.LIMIT_NEWS)
                all_items.extend(rich_news)
            except Exception as e:
                print(f"   ❌ News Fetch Failed: {e}")

        print(f"   ✅ Total Items: {len(all_items)}")
        return all_items

    def analyze_data(self, all_items):
        if not all_items: return []

        print(f"--- 🧠 Analyzing {len(all_items)} items with AI ---")
        
        # EXTRACT TEXT ONLY for the AI
        # (The AI doesn't need the image URL or date)
        texts_only = [item['text'] for item in all_items]
        
        ai_results = []
        for i in range(0, len(texts_only), self.BATCH_SIZE):
            batch = texts_only[i : i + self.BATCH_SIZE]
            results = analyze_batch_with_retry(batch, batch_index=i)
            ai_results.extend(results)
            
        return ai_results

    def merge_and_save(self, all_items, ai_results):
        """
        Merges AI scores into the Rich Data Objects.
        """
        processed_data = []

        for i, item in enumerate(all_items):
            # 1. Get AI Result
            if i < len(ai_results):
                res = ai_results[i]
                if isinstance(res, list): res = res[0]
                
                # Append analysis to the Rich Object
                item['sentiment'] = res.get('sentiment', 'Neutral')
                item['impact'] = res.get('impact', 'Low')
                item['score'] = res.get('score', 0.0)
            else:
                item['score'] = 0.0
                item['sentiment'] = "Neutral"

            item['processed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            processed_data.append(item)

        # 2. Save to files
        sources_map = {
            "Twitter": "final_tweets.json",
            "News_Media": "final_news.json"
        }

        saved_counts = {"Twitter": 0, "News": 0}

        for source_key, filename in sources_map.items():
            subset = [x for x in processed_data if x['source'] == source_key]
            
            if subset:
                with open(filename, "w", encoding='utf-8') as f:
                    json.dump(subset, f, indent=4)
                saved_counts[source_key] = len(subset)

        print(f"--- 💾 Saved: {saved_counts['Twitter']} Tweets, {saved_counts['News']} News ---")
        return saved_counts

    def run_pipeline(self, sources=["twitter", "news"]):
        data = self.fetch_data(sources)
        if not data: return
        results = self.analyze_data(data)
        self.merge_and_save(data, results)
        print(f"✅ Pipeline Finished.")

    # (start_auto_refresh remains the same as before)