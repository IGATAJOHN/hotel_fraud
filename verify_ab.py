import urllib.request
import json
import time
import urllib.error

def call_api(url, method="GET", data=None):
    if data:
        data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            raw = res.read().decode()
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code}: {e.read().decode()}")
        raise e
    except Exception as e:
        print(f"FAILED TO CALL API: {e}")
        raise e

def verify_ab_testing():
    base_url = "http://127.0.0.1:8000"
    
    print("--- Verifying Initial State (100% v1) ---")
    reports = []
    for _ in range(5):
        res = call_api(f"{base_url}/score", "POST", {"data": {"lead_time": 10, "booking_hour": 14}})
        reports.append(res["model_info"]["version"])
    print(f"Versions seen: {reports}")

    print("\n--- Enabling A/B Test (50/50 split) ---")
    call_api(f"{base_url}/admin/models/split", "POST", {"service": "fraud", "split": {"v1": 0.5, "v2": 0.5}})
    
    reports = []
    for _ in range(20):
        res = call_api(f"{base_url}/score", "POST", {"data": {"lead_time": 10, "booking_hour": 14}})
        reports.append(res["model_info"]["version"])
    
    v1_count = reports.count("v1")
    v2_count = reports.count("v2")
    print(f"Versions seen in A/B test: v1={v1_count}, v2={v2_count}")

    print("\n--- Verifying Rollback (Back to 100% v1) ---")
    call_api(f"{base_url}/admin/models/rollback", "POST", {"service": "fraud", "target_version": "v1"})
    
    reports = []
    for _ in range(5):
        res = call_api(f"{base_url}/score", "POST", {"data": {"lead_time": 10, "booking_hour": 14}})
        reports.append(res["model_info"]["version"])
    print(f"Versions seen after rollback: {reports}")

if __name__ == "__main__":
    verify_ab_testing()
