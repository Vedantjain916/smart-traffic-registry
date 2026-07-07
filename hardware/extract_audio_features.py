import os
import numpy as np
import pandas as pd
import librosa
import warnings

# Suppress librosa load warnings
warnings.filterwarnings('ignore')

# =====================================================
# PROJECT ROOT & PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define your input folder where you have folders like 'car/', 'bus/', 'truck/'
WAV_ROOT = os.path.join(BASE_DIR, "processed", "wav_audio")
OUTPUT_ROOT = os.path.join(BASE_DIR, "processed", "datasets")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=8000, mono=True)
    
    if len(y) == 0:
        raise ValueError("Empty audio file")

    y = y - np.mean(y) # Remove DC offset

    # Audio Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    features = {f"MFCC_{i}": float(mfcc_mean[i]) for i in range(13)}
    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features["rms_energy"] = float(np.mean(librosa.feature.rms(y=y)))
    features["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # VIBRATION PADDING (Required to match your Master Dataset Schema)
    features["vib_mean"] = 0.0
    features["vib_std"] = 0.0
    features["vib_rms"] = 0.0
    features["vib_max"] = 0.0
    features["vib_min"] = 0.0
    features["vib_peak_to_peak"] = 0.0
    
    return features

# =====================================================
# MAIN
# =====================================================
def main():
    print("\n--- STARTING EXTRACTOR (WITH VIBRATION PADDING) ---")
    
    if not os.path.exists(WAV_ROOT):
        print(f"Error: Folder not found: {WAV_ROOT}")
        return

    # Process every subfolder as a label (e.g., 'car', 'bus')
    subfolders = [f for f in os.listdir(WAV_ROOT) if os.path.isdir(os.path.join(WAV_ROOT, f))]

    for label in subfolders:
        folder_path = os.path.join(WAV_ROOT, label)
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        
        print(f"\nProcessing Label: {label} ({len(wav_files)} files)")
        
        data = []
        for file in wav_files:
            try:
                features = extract_features(os.path.join(folder_path, file))
                features["file_name"] = file
                features["folder"] = "public_dataset"
                features["label"] = label
                data.append(features)
            except Exception as e:
                print(f"Failed {file}: {e}")

        if data:
            df = pd.DataFrame(data)
            # Reorder columns to match Master Dataset exactly
            cols = ['MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 
                    'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'spectral_centroid', 
                    'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy', 
                    'spectral_flatness', 'file_name', 'vib_mean', 'vib_std', 'vib_rms', 'vib_max', 
                    'vib_min', 'vib_peak_to_peak', 'folder', 'label']
            
            df = df[cols]
            
            output_file = os.path.join(OUTPUT_ROOT, f"master_dataset_{label}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()