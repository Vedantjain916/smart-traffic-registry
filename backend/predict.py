import librosa
import numpy as np
import pandas as pd
import scipy.signal
import os
from model_loader import get_yamnet, get_xgb, get_encoder, get_scaler

def extract_all_features(audio_path):
    """
    Exactly the same feature extraction as in scripts/3_extract_features.py
    """
    # Load audio (YAMNet requires 16000Hz Mono)
    wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(wav_data) == 0:
        raise ValueError("Empty audio file")
    
    # Pad very short audio to at least 0.5 seconds to avoid issues with feature extraction
    min_samples = sr // 2  # 0.5 seconds
    if len(wav_data) < min_samples:
        wav_data = np.pad(wav_data, (0, min_samples - len(wav_data)), mode='constant')

    wav_data = wav_data / np.max(np.abs(wav_data))  # Normalize

    features = {}

    # 1. YAMNet Features (Using Mean AND Std to capture temporal variance)
    yamnet = get_yamnet()
    scores, embeddings, spectrogram = yamnet(wav_data)
    embed_numpy = embeddings.numpy()
    
    avg_embedding = np.mean(embed_numpy, axis=0)
    
    for i in range(1024):
        features[f"YAMNET_mean_{i}"] = float(avg_embedding[i])

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

def process_audio_prediction(file_path):
    """
    ROUTE 1: Web UI Uploads
    Processes raw .wav files or .raw files uploaded from the React frontend.
    """
    xgb_model = get_xgb()
    encoder = get_encoder()
    scaler = get_scaler()

    try:
        # Check if it's a raw audio file
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.raw':
            # Load raw audio (assuming 8000Hz, 16-bit signed PCM, mono, little-endian)
            raw_sr = 8000
            raw_data = np.fromfile(file_path, dtype=np.int16)
            wav_data = raw_data.astype(np.float32) / np.iinfo(np.int16).max
            wav_data = librosa.resample(wav_data, orig_sr=raw_sr, target_sr=16000)
            
            # Create a temporary WAV file to pass to extract_all_features
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, wav_data, 16000)
                file_path = tmp.name
        
        # Extract all features using the exact training code
        features = extract_all_features(file_path)
        
        # 5. Model Inference
        # Force the dataframe to have the exact column order the XGBoost model expects
        df_features = pd.DataFrame([features])[list(xgb_model.feature_names_in_)]
        
        # Apply scaling
        df_features_scaled = scaler.transform(df_features)

        probabilities = xgb_model.predict_proba(df_features_scaled)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx] * 100)

        # Build Top 3 for the frontend UI
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3 = [
            {"label": encoder.inverse_transform([idx])[0], "confidence": float(probabilities[idx] * 100)}
            for idx in top_3_indices
        ]

        return {"success": True, "prediction": predicted_label, "confidence": confidence, "top_3": top_3}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}


def process_live_prediction(payload_dict):
    """
    ROUTE 2: IoT Hardware 
    Bypasses audio extraction. Accepts a pre-calculated JSON dictionary 
    directly from the ESP8266/NodeMCU on the street.
    """
    xgb_model = get_xgb()
    encoder = get_encoder()
    scaler = get_scaler()

    try:
        # Convert incoming JSON dictionary to a DataFrame
        df_features = pd.DataFrame([payload_dict])
        
        # Ensure it has all required columns, fill missing ones with 0.0 just in case
        for col in xgb_model.feature_names_in_:
            if col not in df_features.columns:
                df_features[col] = 0.0
                
        df_features = df_features[list(xgb_model.feature_names_in_)]
        
        # Apply scaling
        df_features_scaled = scaler.transform(df_features)

        probabilities = xgb_model.predict_proba(df_features_scaled)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx] * 100)

        return {"success": True, "prediction": predicted_label, "confidence": confidence}

    except Exception as e:
        return {"success": False, "error": f"Live prediction failed: {str(e)}"}
