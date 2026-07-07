import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your ML loader and prediction logic
from model_loader import load_all_models
from predict import process_audio_prediction

app = Flask(__name__)
# Totally permissive CORS for local development
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load machine learning models into memory exactly once on startup
load_all_models()

# --- Global memory to store the last 50 live predictions ---
prediction_history = []

# =====================================================
# ROUTE 1: The ESP32 & Web UI Endpoint (Audio + Vibration)
# =====================================================
@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if file and (file.filename.lower().endswith('.wav') or file.filename.lower().endswith('.raw')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 1. Capture real ADXL345 data from the ESP32 headers
            # (If uploaded from Web UI, these will safely default to 0.0)
            real_vib_x = request.headers.get("X-Vibration-X", "0.0")
            real_vib_y = request.headers.get("X-Vibration-Y", "0.0")
            real_vib_z = request.headers.get("X-Vibration-Z", "0.0")

            # 2. Run full heavy audio extraction (Using your existing predict.py)
            result = process_audio_prediction(filepath)

            # 3. Cleanup file
            if os.path.exists(filepath):
                os.remove(filepath)

            # 4. Save successful predictions to history for the React Dashboard
            if result.get('success'):
                history_entry = {
                    "id": len(prediction_history) + 1,
                    "timestamp": datetime.now().strftime("%I:%M:%S %p"),
                    "vehicle": result['prediction'].upper(),
                    "confidence": result['confidence'],
                    "adxl_x": float(real_vib_x),
                    "adxl_y": float(real_vib_y),
                    "adxl_z": float(real_vib_z)
                }
                prediction_history.insert(0, history_entry) # Put newest at the top
                
                # Keep only the last 50 entries so server memory doesn't blow up
                if len(prediction_history) > 50:
                    prediction_history.pop()

            return jsonify(result), 200 if result.get('success') else 500
            
        return jsonify({"success": False, "error": "Invalid file type. Please upload a .wav or .raw"}), 400
    except Exception as e:
        print(f"Error in predict_endpoint: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# =====================================================
# ROUTE 2: Frontend Dashboard Stream
# =====================================================
@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns the list of recent live predictions to the React app"""
    return jsonify({"success": True, "data": prediction_history})


if __name__ == '__main__':
    print("Traffic ML Server running on http://0.0.0.0:5000")
    # 0.0.0.0 is CRITICAL so the ESP32 can connect over Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)