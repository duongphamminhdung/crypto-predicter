import requests

# --- CONFIGURATION ---
# 1. GET YOUR KEY AT: rapidapi.com 
RAPID_API_KEY = "6fe364c3a5msh6707fe2be4cbc8bp12d058jsna848daf6c696"
RAPID_API_HOST = "twitter-api45.p.rapidapi.com" 

def fetch_real_tweets(query="Bitcoin", limit=10):
    print(f"   [RapidAPI] Fetching RICH tweets for '{query}'...")
    
    url = f"https://{RAPID_API_HOST}/search.php"
    
    if "min_faves" not in query:
        final_query = f"{query} lang:en min_faves:5000 -is:retweet"
    else:
        final_query = query
        
    querystring = {
        "query": final_query, 
        "limit": str(limit)
    }
    
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": RAPID_API_HOST
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        
        rich_tweets = []
        
        # Parse Timeline
        if 'timeline' in data:
            for item in data['timeline']:
                # 1. Check for Text (Skip if empty/media only)
                text = item.get('text', '').strip()
                if not text: 
                    continue
                
                # 2. Extract Author Info (Try multiple locations)
                user_info = item.get('user_info', {}) 
                if not user_info:
                    user_info = item.get('user', {})

                # 3. Get Author Handle (Critical for the image hack)
                screen_name = user_info.get('screen_name', '')
                if not screen_name:
                    # Some APIs put it directly in the item
                    screen_name = item.get('screen_name', 'unknown_user')

                # 4. THE IMAGE FIX: Use Unavatar.io
                # We try the API first. If it's empty, we use the public proxy.
                api_avatar = user_info.get('profile_image_url_https') or user_info.get('profile_image_url')
                
                if api_avatar:
                    avatar_url = api_avatar.replace("_normal", "") # High-res
                elif screen_name and screen_name != 'unknown_user':
                    # FALLBACK: Use free public avatar service
                    avatar_url = f"https://unavatar.io/twitter/{screen_name}"
                else:
                    # Final fallback: A generic grey ghost image
                    avatar_url = "https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"

                # 5. Get Date
                date_posted = item.get('created_at', 'Unknown Date')

                # 6. Build Rich Object
                rich_tweets.append({
                    "text": text,
                    "author_name": user_info.get('name', screen_name),
                    "author_handle": screen_name,
                    "author_image": avatar_url, # <--- Now guaranteed to have a URL
                    "date": date_posted,
                    "source": "Twitter",
                    "url": f"https://x.com/{screen_name}/status/{item.get('tweet_id')}"
                })
        
        print(f"   ✅ Found {len(rich_tweets)} rich tweets.")
        return rich_tweets

    except Exception as e:
        print(f"   ❌ Twitter API Error: {e}")
        return []