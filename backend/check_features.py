
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
XGB_PATH = os.path.join(MODEL_DIR, "vehicle_classifier_xgb.pkl")

xgb_model = joblib.load(XGB_PATH)
print(f"Model expects {len(xgb_model.feature_names_in_)} features:")
print(xgb_model.feature_names_in_)
