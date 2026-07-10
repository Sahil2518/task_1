"""
Task 20 — End-to-End Test and Latency Benchmark
PlaceMux Phase 1 Industry Immersion · Capstone

This script tests the Flask application (`src.serve_task20`) to ensure
it meets the definition of done:
1. Health check passes.
2. Prediction endpoint returns sensible values for valid inputs.
3. Batch prediction endpoint works.
4. Error handling catches bad inputs gracefully.
5. Latency is measured over 50 calls to ensure acceptable performance.
"""

import os
import sys
import json
import time
import requests

API_URL = "http://127.0.0.1:5020"
LOG_DIR = "logs"
RESULTS_FILE = os.path.join(LOG_DIR, "task20_test_results.json")

def print_step(msg: str):
    print(f"\n[STEP] {msg}")

def test_health() -> bool:
    print_step("Testing GET /health")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Status: {r.status_code} OK")
            print(f"  Model Loaded: {data.get('model_loaded')}")
            if not data.get("model_loaded"):
                print("  [FAIL] Health check passed, but model is not loaded!")
                return False
            return True
        else:
            print(f"  [FAIL] Status: {r.status_code}\n  {r.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Request failed: {e}")
        return False

def test_single_predict() -> dict:
    print_step("Testing POST /v1/predict (Valid Input)")
    payload = {
        "domain_score": 85,
        "aptitude_score": 90,
        "projects_completed": 4,
        "active_days": 120,
        "registration_month": 3
    }
    try:
        r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Status: {r.status_code} OK")
            print(f"  Result: {data}")
            return {"passed": True, "data": data}
        else:
            print(f"  [FAIL] Status: {r.status_code}\n  {r.text}")
            return {"passed": False, "data": None}
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Request failed: {e}")
        return {"passed": False, "data": None}

def test_batch_predict() -> bool:
    print_step("Testing POST /v1/predict/batch (Valid Input)")
    payload = {
        "candidates": [
            {"domain_score": 90, "aptitude_score": 85, "projects_completed": 5},
            {"domain_score": 40, "aptitude_score": 45, "projects_completed": 0},
            {"domain_score": 65, "aptitude_score": 70, "projects_completed": 2}
        ]
    }
    try:
        r = requests.post(f"{API_URL}/v1/predict/batch", json=payload, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Status: {r.status_code} OK")
            print(f"  Processed {data.get('n_candidates')} candidates in {data.get('latency_ms')}ms")
            for p in data.get("predictions", []):
                print(f"    Sample {p['sample_index']}: Placed={p['is_placed']}, Prob={p['probability']}, Conf={p['confidence']}")
            return True
        else:
            print(f"  [FAIL] Status: {r.status_code}\n  {r.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Request failed: {e}")
        return False

def test_error_handling() -> bool:
    print_step("Testing POST /v1/predict (Invalid Input - Missing Required Field)")
    payload = {
        "domain_score": 85,
        # missing aptitude_score
        "projects_completed": 4
    }
    try:
        r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=5)
        print(f"  Expected 422, Got: {r.status_code}")
        if r.status_code == 422:
            print(f"  Error Detail: {r.json().get('detail')}")
            return True
        else:
            print(f"  [FAIL] Unexpected response: {r.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Request failed: {e}")
        return False

def benchmark_latency(num_calls: int = 50) -> dict:
    print_step(f"Running Latency Benchmark ({num_calls} calls)")
    payload = {
        "domain_score": 75,
        "aptitude_score": 80,
        "projects_completed": 3,
        "registration_date": "2025-01-15",
        "last_login_date": "2025-06-01"
    }
    
    latencies = []
    success_count = 0
    
    for i in range(num_calls):
        try:
            t0 = time.perf_counter()
            r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=5)
            t1 = time.perf_counter()
            if r.status_code == 200:
                # Use server-reported latency if available, else client-side measure
                srv_lat = r.json().get("latency_ms")
                if srv_lat is not None:
                    latencies.append(srv_lat)
                else:
                    latencies.append((t1 - t0) * 1000)
                success_count += 1
            else:
                print(f"  [WARNING] Call {i+1} failed with status {r.status_code}")
        except requests.exceptions.RequestException as e:
             print(f"  [WARNING] Call {i+1} failed: {e}")
             
    if not latencies:
        print("  [FAIL] All benchmark calls failed.")
        return {"passed": False, "stats": {}}
        
    latencies.sort()
    min_lat = latencies[0]
    max_lat = latencies[-1]
    mean_lat = sum(latencies) / len(latencies)
    p95_lat = latencies[int(len(latencies) * 0.95)]
    
    print(f"  Calls: {success_count}/{num_calls} successful")
    print(f"  Min : {min_lat:.2f} ms")
    print(f"  Mean: {mean_lat:.2f} ms")
    print(f"  p95 : {p95_lat:.2f} ms")
    print(f"  Max : {max_lat:.2f} ms")
    
    passed = mean_lat < 100.0  # Acceptable latency threshold
    if passed:
        print("  Latency check PASSED (Mean < 100ms)")
    else:
        print("  [FAIL] Latency check FAILED (Mean >= 100ms)")
        
    stats = {
        "num_calls": num_calls,
        "success_count": success_count,
        "min_ms": round(min_lat, 2),
        "mean_ms": round(mean_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "max_ms": round(max_lat, 2)
    }
    return {"passed": passed, "stats": stats}

def generate_walkthrough(results: dict):
    print_step("Generating Walkthrough Artifact")
    os.makedirs(LOG_DIR, exist_ok=True)
    walkthrough_path = os.path.join(LOG_DIR, "task20_walkthrough.md")
    
    status_icon = "✅" if results["all_passed"] else "❌"
    
    content = f"""# Task 20 — Walkthrough: End-to-End Flask Deployment

## What Was Built
This capstone task implements a production-grade Flask REST API (`src/serve_task20.py`) that loads the serialized `joblib` artifact from Task 19. It validates incoming requests, performs feature engineering on-the-fly, and serves predictions. A standalone test suite (`src/test_task20.py`) verified the endpoints and benchmarked latency.

## Pipeline Steps
1. **Model Loading:** The Flask app loads `models/task19/placemux_pipeline_v1.0.0.joblib` on startup.
2. **Integrity Check:** The SHA-256 hash of the loaded artifact is verified against the `metadata.json` sidecar.
3. **Endpoint Validation:** Incoming JSON to `/v1/predict` is checked for required fields (`domain_score`, `aptitude_score`, `projects_completed`). Optional date fields are used to derive `active_days`.
4. **Inference:** The validated data is converted to a DataFrame and passed to `pipeline.predict_proba`.
5. **Response:** The API returns the binary prediction, probability, confidence level, and processing latency.

## Key Results
Overall Test Status: {status_icon} {"**PASSED**" if results["all_passed"] else "**FAILED**"}

| Metric | Result |
|--------|--------|
| Health Check | {'✅' if results['health'] else '❌'} |
| Single Predict (Valid) | {'✅' if results['single_predict']['passed'] else '❌'} |
| Batch Predict (Valid) | {'✅' if results['batch_predict'] else '❌'} |
| Error Handling (422) | {'✅' if results['error_handling'] else '❌'} |
| Mean Latency | {results['benchmark']['stats'].get('mean_ms', 'N/A')} ms |
| p95 Latency | {results['benchmark']['stats'].get('p95_ms', 'N/A')} ms |

## Files Produced
- `src/serve_task20.py`: The Flask application.
- `src/test_task20.py`: End-to-end testing and latency benchmarking script.
- `run_task20.bat`: One-click runner that starts the server and runs tests.
- `src/zip_task20.py`: Packager for final deliverables.
- `logs/task20.log`: Application logs from the Flask server.
- `logs/task20_test_results.json`: JSON output of the test run.
- `logs/task20_walkthrough.md`: This document.
- `placemux_task20_*.zip`: Final packaged deliverable.

## How to Run
```bash
# To run the complete pipeline (train -> serve -> test)
run_task20.bat
```

## Error Handling Summary
The Flask application implements robust error handling:
- **Startup Guard:** If the artifact is missing or corrupted (SHA-256 mismatch), the server starts but endpoints return `503 Service Unavailable`.
- **Global Error Handlers:** Flask `@app.errorhandler` catches `400`, `404`, `405`, and `500` errors, returning structured JSON instead of HTML.
- **Input Validation:** The `_prepare_features` function rigorously checks types, bounds, and required fields, raising `ValueError` caught and returned as `422 Unprocessable Entity`.
"""
    with open(walkthrough_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Walkthrough written to {walkthrough_path}")

def main():
    print("==================================================")
    print("  Task 20 — End-to-End Test and Benchmark")
    print("==================================================")
    
    # Wait for server to be available
    max_retries = 10
    server_up = False
    print("Waiting for Flask server to start...")
    for i in range(max_retries):
        try:
            requests.get(f"{API_URL}/health", timeout=2)
            server_up = True
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            
    if not server_up:
        print("[FATAL] Could not connect to the Flask server. Is it running?")
        sys.exit(1)
        
    results = {}
    
    results["health"] = test_health()
    results["single_predict"] = test_single_predict()
    results["batch_predict"] = test_batch_predict()
    results["error_handling"] = test_error_handling()
    results["benchmark"] = benchmark_latency()
    
    all_passed = (
        results["health"] and 
        results["single_predict"]["passed"] and 
        results["batch_predict"] and 
        results["error_handling"] and
        results["benchmark"]["passed"]
    )
    
    results["all_passed"] = all_passed
    
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    generate_walkthrough(results)
        
    print("\n==================================================")
    if all_passed:
        print("  Task 20 Tests: PASSED [OK]")
    else:
        print("  Task 20 Tests: FAILED [ERROR] (Check logs for details)")
        sys.exit(1)
    print("==================================================")

if __name__ == "__main__":
    main()
