import time
from backend_service import CryptoSentimentService

def test_manual_run():
    print("\n--- 🧪 TEST 1: Manual Run (Twitter Only) ---")
    service = CryptoSentimentService()
    
    # We only ask for 'twitter' to make the test fast
    service.run_pipeline(sources=["twitter"])
    print("✅ Manual Run Complete. Check 'final_tweets.json'.")

def test_full_pipeline():
    print("\n--- 🧪 TEST 2: Full Pipeline (All Sources) ---")
    service = CryptoSentimentService()
    
    # Analyze everything
    service.run_pipeline(sources=["twitter", "news"])
    print("✅ Full Run Complete. Check all 2 JSON files.")

def test_auto_refresh():
    print("\n--- 🧪 TEST 3: Auto-Refresh System ---")
    service = CryptoSentimentService()
    
    # Set interval to 0.001 hours (about 3 seconds) just for this test!
    # In real life, you would use 3, 6, 12.
    print("   (Simulating a 3-hour refresh cycle... but fast)")
    service.start_auto_refresh(interval_hours=0.001, sources=["news"])
    
    # Keep the script alive for 10 seconds to let the background job run twice
    try:
        for i in range(10):
            print(f"   websiteserver.exe is running... {i+1}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    print("✅ Auto-Refresh Test Complete.")

if __name__ == "__main__":
    # Uncomment the one you want to test
    
    # 1. Run this first to see if basic fetching works
    #test_manual_run()
    
    # 2. Run this to test everything
    test_full_pipeline()
    
    # 3. Run this to see the background timer working
    #test_auto_refresh()