import requests
import logging
from django.conf import settings
from django.core.cache import cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TwitterService:
    """Fetch crypto-related tweets from Twitter API v2"""
    
    def __init__(self):
        self.bearer_token = settings.TWITTER_BEARER_TOKEN
        self.base_url = "https://api.twitter.com/2/tweets/search/recent"
        self.headers = self._get_headers()
        self.rate_limit_reset = None
    
    def _get_headers(self):
        """Create authorization headers"""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "CryptoPredicter/1.0"
        }
    
    def fetch_crypto_news(self, keywords=None, max_results=10):
        """
        Fetch recent crypto news tweets with fallback to cache
        
        Args:
            keywords: List of keywords to search (default: crypto, bitcoin, ethereum)
            max_results: Number of tweets to fetch (10-100, default: 10)
        
        Returns:
            List of tweet dictionaries or cached tweets on error
        """
        
        cache_key = 'twitter_crypto_news'
        
        # VALIDATE max_results parameter
        if max_results < 10:
            logger.warning(f"max_results {max_results} is too low, using minimum of 10")
            max_results = 10
        elif max_results > 100:
            logger.warning(f"max_results {max_results} is too high, using maximum of 100")
            max_results = 100

        cached_news = cache.get(cache_key)
        return cached_news if cached_news else self._get_placeholder_tweets()

        # If we hit rate limit, use cache indefinitely
        if self.rate_limit_reset and datetime.now() < self.rate_limit_reset:
            cached_news = cache.get(cache_key)
            if cached_news:
                logger.warning("Using cached tweets due to rate limit")
                return cached_news
            else:
                logger.error("Rate limited and no cache available")
                return self._get_placeholder_tweets()
        
        # Check cache first (10 minute TTL - respects rate limits)
        cached_news = cache.get(cache_key)
        if cached_news:
            logger.info("Returning cached tweets")
            return cached_news
        
        if not self.bearer_token:
            logger.error("Twitter Bearer Token not configured")
            return self._get_placeholder_tweets()
        
        if keywords is None:
            keywords = ['crypto', 'bitcoin', 'ethereum', 'BTC', 'ETH']
        
        # Build search query
        query = self._build_query(keywords)
        
        params = {
            'query': query,
            'max_results': max_results,  # Now guaranteed to be 10-100
            'tweet.fields': 'created_at,public_metrics,author_id',
            'expansions': 'author_id',
            'user.fields': 'username,name,profile_image_url,verified'
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=5
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                reset_time = response.headers.get('x-rate-limit-reset')
                if reset_time:
                    self.rate_limit_reset = datetime.fromtimestamp(int(reset_time))
                    logger.error(f"Rate limited until {self.rate_limit_reset}")
                
                # Return cached data if available
                if cached_news:
                    return cached_news
                return self._get_placeholder_tweets()
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check if API returned an error
            if 'errors' in data:
                logger.error(f"Twitter API error: {data['errors']}")
                return cached_news if cached_news else self._get_placeholder_tweets()
            
            tweets = self._parse_response(data)
            
            if not tweets:
                logger.warning("No tweets returned from API")
                return cached_news if cached_news else self._get_placeholder_tweets()
            
            # Cache results for 10 minutes
            cache.set(cache_key, tweets, 600)
            logger.info(f"Cached {len(tweets)} tweets")
            
            return tweets
            
        except requests.exceptions.Timeout:
            logger.error("Twitter API request timeout")
            return cached_news if cached_news else self._get_placeholder_tweets()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Twitter API HTTP error: {e.response.status_code} - {e.response.text}")
            return cached_news if cached_news else self._get_placeholder_tweets()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Twitter API connection error: {e}")
            return cached_news if cached_news else self._get_placeholder_tweets()
            
        except Exception as e:
            logger.error(f"Unexpected error fetching tweets: {e}", exc_info=True)
            return cached_news if cached_news else self._get_placeholder_tweets()
    
    def _build_query(self, keywords):
        """Build Twitter search query"""
        keyword_query = ' OR '.join(keywords)
        return f"({keyword_query}) -is:retweet -is:reply lang:en"
    
    def _parse_response(self, data):
        """Parse Twitter API response into clean format"""
        tweets = []
        
        if 'data' not in data:
            logger.warning("No 'data' field in API response")
            return tweets
        
        # Create user lookup dictionary
        users = {}
        if 'includes' in data and 'users' in data['includes']:
            for user in data['includes']['users']:
                users[user['id']] = user
        
        # Parse tweets
        for tweet in data['data']:
            try:
                user = users.get(tweet['author_id'], {})
                
                tweets.append({
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'created_at': tweet.get('created_at', ''),
                    'author_name': user.get('name', 'Unknown'),
                    'author_username': user.get('username', 'unknown'),
                    'author_image': user.get('profile_image_url', ''),
                    'author_verified': user.get('verified', False),
                    'likes': tweet.get('public_metrics', {}).get('like_count', 0),
                    'retweets': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    'replies': tweet.get('public_metrics', {}).get('reply_count', 0),
                    'url': f"https://twitter.com/{user.get('username', '')}/status/{tweet['id']}",
                    'sentiment': self.get_sentiment(tweet['text'])
                })
            except Exception as e:
                logger.warning(f"Error parsing tweet: {e}")
                continue
        
        return tweets
    
    def _get_placeholder_tweets(self):
        """Return placeholder tweets when API fails"""
        logger.info("Returning placeholder tweets")
        return [
            {
                'id': '0',
                'text': '🚨 Unable to fetch live crypto news. Please check Twitter API connection.',
                'created_at': datetime.now().isoformat(),
                'author_name': 'System',
                'author_username': 'system',
                'author_image': '',
                'author_verified': False,
                'likes': 0,
                'retweets': 0,
                'replies': 0,
                'url': '#',
                'sentiment': 'neutral'
            }
        ]
    
    def get_sentiment(self, text):
        """Simple sentiment analysis based on keywords"""
        positive_words = ['moon', 'bullish', 'pump', 'breakout', 'surge', 'gain', 'rally', 'spike']
        negative_words = ['crash', 'bearish', 'dump', 'fall', 'loss', 'decline', 'drop', 'bear']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'