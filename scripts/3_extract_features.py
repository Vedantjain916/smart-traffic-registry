
import os
import numpy as np
import pandas as pd
import librosa
import scipy.signal
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_hub as hub

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_ROOT = os.path.join(BASE_DIR, "processed", "wav_audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The master dataset file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_audio_dataset.csv")

# Checkpoint file to track processed files
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "extraction_checkpoint.json")

BATCH_SIZE = 100  # Process 100 files per batch

LABEL_NAME_MAP = {
    "car": "Car",
    "bus": "Bus",
    "bike": "Bike",
    "truck": "Truck",
    "background": "Background",
}

# =====================================================
# LOAD YAMNET
# =====================================================
print("Loading Google YAMNet Model...")
try:
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet Loaded Successfully!\n")
except Exception as e:
    print(f"Failed to load YAMNet: {e}")
    exit()

# =====================================================
# EXTRACTION ENGINE (Audio + Synthetic Vibration + Temporal Variance)
# =====================================================
def extract_all_features(audio_path):
    # Load audio (YAMNet requires 16000Hz Mono)
    wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(wav_data) == 0:
        raise ValueError("Empty audio file")
    
    # Pad very short audio to at least 0.5 seconds to avoid issues with feature extraction
    min_samples = sr // 2  # 0.5 seconds
    if len(wav_data) < min_samples:
        wav_data = np.pad(wav_data, (0, min_samples - len(wav_data)), mode='constant')

    wav_data = wav_data / np.max(np.abs(wav_data)) # Normalize

    features = {}

    # 1. YAMNet Features (Using Mean AND Std to capture temporal variance)
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    embed_numpy = embeddings.numpy()
    
    avg_embedding = np.mean(embed_numpy, axis=0)
    std_embedding = np.std(embed_numpy, axis=0) # Captures how much the sound changes
    
    for i in range(1024):
        features[f"YAMNET_mean_{i}"] = float(avg_embedding[i])
        # We are omitting std for all 1024 to prevent dataset bloat; focusing on acoustic variance below.

    # 2. Acoustic Features (Librosa) - Adding Deltas and Std Dev
    mfccs = librosa.feature.mfcc(y=wav_data, sr=sr, n_mfcc=13)
    
    # Handle delta extraction with fallback for very short signals
    try:
        mfcc_delta = librosa.feature.delta(mfccs)          # 1st Derivative (Velocity)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2) # 2nd Derivative (Acceleration)
    except:
        mfcc_delta = np.zeros_like(mfccs)
        mfcc_delta2 = np.zeros_like(mfccs)

    for i in range(13):
        features[f"MFCC_mean_{i}"] = float(np.mean(mfccs[i]))
        features[f"MFCC_std_{i}"] = float(np.std(mfccs[i]))               
        features[f"MFCC_delta_mean_{i}"] = float(np.mean(mfcc_delta[i]))  
        features[f"MFCC_delta2_mean_{i}"] = float(np.mean(mfcc_delta2[i]))

    # Adding variance to spectral features
    cent = librosa.feature.spectral_centroid(y=wav_data, sr=sr)
    features['spectral_centroid_mean'] = float(np.mean(cent))
    features['spectral_centroid_std'] = float(np.std(cent)) 
    
    rolloff = librosa.feature.spectral_rolloff(y=wav_data, sr=sr)
    features['spectral_rolloff_mean'] = float(np.mean(rolloff))
    features['spectral_rolloff_std'] = float(np.std(rolloff)) 

    features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=wav_data, sr=sr)))
    features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(wav_data)))
    
    rms = librosa.feature.rms(y=wav_data)
    features['rms_energy_mean'] = float(np.mean(rms))
    features['rms_energy_std'] = float(np.std(rms)) 
    
    features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=wav_data)))

    # 3. Synthetic Vibration Engine (Low-pass filter under 100Hz)
    sos = scipy.signal.butter(4, 100, 'low', fs=sr, output='sos')
    rumble_data = scipy.signal.sosfilt(sos, wav_data)
    
    features["vib_mean"] = float(np.mean(np.abs(rumble_data)))
    features["vib_std"] = float(np.std(rumble_data))
    features["vib_rms"] = float(np.sqrt(np.mean(rumble_data**2)))
    features["vib_max"] = float(np.max(rumble_data))
    features["vib_min"] = float(np.min(rumble_data))
    features["vib_peak_to_peak"] = features["vib_max"] - features["vib_min"]

    return features

# =====================================================
# CHECKPOINT MANAGEMENT
# =====================================================
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_files):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(processed_files), f)

def append_batch_to_dataset(df_batch):
    if os.path.exists(OUTPUT_FILE):
        df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_batch.to_csv(OUTPUT_FILE, index=False)

# =====================================================
# MAIN LOOP WITH BATCH PROCESSING & CHECKPOINTING
# =====================================================
def main():
    print("=" * 50)
    print("AUDIO & VIBRATION FEATURE EXTRACTION")
    print("=" * 50)
    
    if not os.path.exists(WAV_ROOT):
        print(f"Error: Folder not found: {WAV_ROOT}")
        return

    # --- Load checkpoint ---
    processed_files = load_checkpoint()
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing dataset at {OUTPUT_FILE}")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, usecols=['file_name'])
            processed_files.update(existing_df['file_name'].tolist())
        except Exception as e:
            print(f"Warning: Could not load existing file names: {e}")
        print(f"Loaded {len(processed_files)} previously processed files (from checkpoint and dataset)\n")
    else:
        print("No existing dataset or checkpoint found. Starting fresh!\n")

    # Collect all files to process
    valid_folders = {"car", "bus", "bike", "truck", "background"}
    all_files_to_process = []
    
    print("Scanning for new files...")
    for label in sorted(valid_folders):
        folder_path = os.path.join(WAV_ROOT, label)
        if not os.path.exists(folder_path):
            continue
            
        normalized_label = LABEL_NAME_MAP.get(label.lower(), label)
        
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(".wav") and f not in processed_files:
                    full_path = os.path.join(root, f)
                    all_files_to_process.append((f, full_path, normalized_label))
    
    total_new_files = len(all_files_to_process)
    if total_new_files == 0:
        print("\n✅ Dataset is already 100% up-to-date! No new files to process!")
        return
        
    print(f"Found {total_new_files} new files to process!")
    print(f"Processing in batches of {BATCH_SIZE} files\n")
    
    # Process in batches
    current_batch = []
    total_processed = 0
    total_batches = 0
    
    for idx, (filename, full_path, label) in enumerate(all_files_to_process):
        try:
            features = extract_all_features(full_path)
            features["file_name"] = filename
            features["label"] = label
            current_batch.append(features)
            processed_files.add(filename)
            total_processed += 1
            
            # Show progress
            if (idx + 1) % 10 == 0:
                print(f"\rProcessed: {idx + 1}/{total_new_files} files ({(idx+1)/total_new_files*100:.1f}%)", end='', flush=True)
            
            # End of batch
            if len(current_batch) >= BATCH_SIZE:
                # Save batch
                df_batch = pd.DataFrame(current_batch)
                append_batch_to_dataset(df_batch)
                save_checkpoint(processed_files)
                
                total_batches += 1
                print(f"\n✅ Batch {total_batches} completed! Saved {len(current_batch)} files")
                current_batch = []
                
        except Exception as e:
            print(f"\n  -> Failed on {filename}: {e}")
    
    # Process remaining files in last batch
    if current_batch:
        df_batch = pd.DataFrame(current_batch)
        append_batch_to_dataset(df_batch)
        save_checkpoint(processed_files)
        total_batches += 1
        print(f"\n✅ Final batch {total_batches} completed! Saved {len(current_batch)} files")
    
    print("\n" + "=" * 50)
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"Total files processed in this session: {total_processed}")
    print(f"Total batches: {total_batches}")
    print(f"Final dataset saved to: {OUTPUT_FILE}")
    print(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()

