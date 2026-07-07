import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_CSV = os.path.join(BASE_DIR, "processed", "datasets", "final_audio_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load just the header (nrows=0)
df = pd.read_csv(FINAL_CSV, nrows=0)

# Get feature names (exclude label and file_name)
possible_label_names = ['label', 'class', 'category', 'target', 'labels']
label_col = next((col for col in df.columns if col.lower() in possible_label_names), None)
file_col = next((col for col in df.columns if 'file' in col.lower() or 'name' in col.lower()), None)

feature_names = [col for col in df.columns if col not in [label_col, file_col]]

# Save feature_names.pkl
feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_names, f)

print(f"✅ Saved feature names to: {feature_names_path}")
print(f"Number of features: {len(feature_names)}")
