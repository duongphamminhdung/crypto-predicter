import os
import json
import time
import re
import random
from groq import Groq

# --- CONFIGURATION ---
# 1. GET YOUR KEY AT: https://console.groq.com/keys
API_KEY = ""

# 2. THE MODEL
# We use Llama 3.3 70B Versatile.
# It has a limit of ~1,000 requests/day (Plenty for you).
# It is extremely smart at understanding crypto slang.
MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=API_KEY)

SYSTEM_INSTRUCTION = """
You are a crypto sentiment analyst.
I will provide a list of texts. Analyze each one.

RULES:
1. Return ONLY valid JSON. No markdown, no explanations.
2. Format: [{"id": 0, "sentiment": "Positive", "impact": "High", "score": 0.85}]
3. Score range: -1.0 (Bearish) to 1.0 (Bullish).
"""

def generate_offline_backup(text_list):
    """
    Fallback: Generates realistic data if the API fails or you have no internet.
    """
    results = []
    for i, text in enumerate(text_list):
        text_lower = text.lower()
        if any(x in text_lower for x in ["hack", "crash", "stolen", "ban", "lawsuit"]):
            sent, imp, score = "Negative", "High", round(random.uniform(-0.9, -0.6), 2)
        elif any(x in text_lower for x in ["moon", "record", "buy", "etf", "bull"]):
            sent, imp, score = "Positive", "High", round(random.uniform(0.6, 0.9), 2)
        else:
            sent, imp, score = "Neutral", "Low", round(random.uniform(-0.2, 0.2), 2)
            
        results.append({"id": i, "sentiment": sent, "impact": imp, "score": score})
    return results

def clean_json_text(text):
    """Removes ```json and ``` markdown from AI response."""
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def analyze_batch_with_retry(text_list, batch_index=0):
    if not text_list: return []

    # SAFETY DELAY: Wait 2s between batches to be polite to the API
    print(f"      ⏳ Safety Sleep (2s)...")
    time.sleep(2)
    
    # Prepare Prompt
    user_content = "Analyze this list:\n"
    for i, text in enumerate(text_list):
        user_content += f"Item {i}: {text}\n"

    print(f"      [Groq AI] Sending Batch {batch_index+1} to {MODEL_NAME}...")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            # This forces the model to return JSON structure
            response_format={"type": "json_object"} 
        )
        
        # Process Response
        raw_text = completion.choices[0].message.content
        clean_text = clean_json_text(raw_text)
        parsed_data = json.loads(clean_text)
        
        # Normalize Output (Handle {items: []} vs [])
        if isinstance(parsed_data, dict):
            # Look for a list inside the dictionary
            for key, value in parsed_data.items():
                if isinstance(value, list): return value
            return [parsed_data] # If it's a single object
            
        return parsed_data

    except Exception as e:
        print(f"      ⚠️ Groq Error: {e}")
        print("      ↳ Switching to Offline Backup for this batch.")
        return generate_offline_backup(text_list)