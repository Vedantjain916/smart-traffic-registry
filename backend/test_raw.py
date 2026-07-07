
import numpy as np
import librosa
import soundfile as sf
from predict import process_audio_prediction
from model_loader import load_all_models

load_all_models()

# Pick a sample wav file
test_wav = r"c:\Users\VIDANT\Desktop\traffic project\processed\wav_audio\wav_audio_7\audio_5s_to_9s.wav"

# Load the wav file
wav_data, sr = librosa.load(test_wav, sr=8000, mono=True)

# Convert to 16-bit PCM
wav_data_int16 = (wav_data * 32767).astype(np.int16)

# Save as raw audio
raw_file = r"c:\Users\VIDANT\Desktop\traffic project\backend\test_audio.raw"
wav_data_int16.tofile(raw_file)

print(f"Raw file created: {raw_file}")

# Test processing this raw file!
result = process_audio_prediction(raw_file)
print(f"Raw file prediction result: {result}")
