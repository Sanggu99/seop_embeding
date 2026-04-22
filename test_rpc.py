import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

# Dummy embedding (3072 zeros for gemini-embedding-001)
payload = {
    "query_embedding": [0.0] * 3072,
    "match_threshold": 0.0,
    "match_count": 5
}

rpc_url = f"{url}/rest/v1/rpc/match_images"
print(f"Testing RPC: {rpc_url}")
try:
    res = requests.post(rpc_url, headers=headers, json=payload, timeout=10)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Error: {e}")
