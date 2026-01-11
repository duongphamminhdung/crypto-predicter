import feedparser
import time
from urllib.parse import urlparse

RSS_URLS = [
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
]

def get_favicon(url):
    """
    Extracts the domain from a URL and asks Google for the Favicon.
    """
    try:
        domain = urlparse(url).netloc
        # Google S2 Service: Free favicons for any domain
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
    except:
        return ""

def fetch_crypto_news(limit=5):
    print(f"   [RSS] Fetching news with Favicons...")
    news_items = []
    
    for url in RSS_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                
                # Get Domain Favicon
                article_url = entry.link
                favicon = get_favicon(article_url)
                
                news_items.append({
                    "text": entry.title,
                    "url": article_url,
                    "source": "News_Media",
                    "favicon": favicon, 
                    "timestamp": time.time(),
                    "date_display": time.strftime('%Y-%m-%d %H:%M', time.localtime())
                })
        except: pass
            
    print(f"   ✅ Found {len(news_items)} headlines.")
    return news_items