
from predict import process_audio_prediction
from model_loader import load_all_models
import os

# Load the models first
load_all_models()

# Pick a sample file from processed/wav_audio
test_file = r"c:\Users\VIDANT\Desktop\traffic project\processed\wav_audio\wav_audio_7\audio_5s_to_9s.wav"

print(f"Testing with file: {test_file}")
print(f"File exists: {os.path.exists(test_file)}")

result = process_audio_prediction(test_file)
print(f"Result: {result}")
