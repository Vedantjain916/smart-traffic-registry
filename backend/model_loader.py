import os
import pickle
import xgboost as xgb
import tensorflow_hub as hub
import warnings

# Force CPU mode to prevent the WSL CUDA crash
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
XGB_PATH = os.path.join(MODEL_DIR, "vehicle_classifier_xgb.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

yamnet_model = None
xgb_model = None
label_encoder = None
scaler = None

def load_all_models():
    global yamnet_model, xgb_model, label_encoder, scaler
    
    if yamnet_model is None:
        print("Loading YAMNet Deep Audio Extractor...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet loaded.")
        
    if xgb_model is None:
        print("Loading Traffic XGBoost Model, Scaler, and Encoder...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(XGB_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("XGBoost, Scaler, and Encoder loaded.")

def get_yamnet(): return yamnet_model
def get_xgb(): return xgb_model
def get_encoder(): return label_encoder
def get_scaler(): return scaler