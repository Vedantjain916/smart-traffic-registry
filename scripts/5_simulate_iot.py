import requests
import time
import json
import random

# Point this to your local Flask backend running in WSL
FLASK_API_URL = "http://localhost:5000/api/predict_live"

print("===========================================")
print("ESP8266 / NODEMCU HARDWARE SIMULATOR")
print("===========================================")
print(f"Targeting Backend: {FLASK_API_URL}\n")

def generate_synthetic_payload():
    """Simulates the math an ESP8266 would do before sending data"""
    # Generating fake MFCCs and synthetic vibration data for a test
    payload = {
        "device_id": "ESP8266_NODE_01",
        "timestamp": time.time(),
        "features": {
            "spectral_centroid": random.uniform(500, 1500),
            "rms_energy": random.uniform(0.01, 0.05),
            "vib_mean": random.uniform(0.5, 2.5),
            "vib_max": random.uniform(1.0, 4.0),
            # In a real scenario, you'd send the full feature array here
        }
    }
    return payload

try:
    while True:
        payload = generate_synthetic_payload()
        print(f"[>] Sending payload from {payload['device_id']}...")
        
        try:
            # Send the simulated HTTP POST request just like an ESP8266 would
            response = requests.post(FLASK_API_URL, json=payload, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                print(f"    [OK] Backend Prediction: {result.get('prediction', 'UNKNOWN').upper()}")
                print(f"    [OK] Confidence: {result.get('confidence', 0):.1f}%")
            else:
                print(f"    [ERROR] Backend returned {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("    [ERROR] Connection refused. Is your Flask app.py running?")
            
        # Wait 3 seconds before the next "sensor reading"
        time.sleep(3)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")