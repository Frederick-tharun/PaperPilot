import time
import requests

API_URL = "http://127.0.0.1:8000"
question = "How do echo state networks handle temporal dependencies?"

print("=== COLD START (models loading) ===")
try:
    start = time.time()
    r = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": 5}, timeout=120)
    total = time.time() - start
    if r.status_code == 200:
        data = r.json()
        print(f"Total latency: {total:.2f}s")
        print(f"Status: {'Refused' if data['refused'] else 'Answered'}")
        print(f"Sources retrieved: {len(data['sources'])}")
    else:
        print(f"Error: {r.status_code}")
except Exception as e:
    print(f"Connection error: {e}")

print("\n=== WARM START (cached models) ===")
try:
    start = time.time()
    r = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": 5}, timeout=120)
    total = time.time() - start
    if r.status_code == 200:
        data = r.json()
        print(f"Total latency: {total:.2f}s")
        print(f"Status: {'Refused' if data['refused'] else 'Answered'}")
        print(f"Answer length: {len(data['answer'])} chars")
except Exception as e:
    print(f"Error: {e}")
