import librosa
import librosa.display
import numpy as np
import pandas as pd
import os

sr = 16000  # TIMIT dataset uses a 16 kHz sample rate

input_dir = "/Users/karim/Desktop/dataset/en/validated_clips"  
output_dir = "/Users/karim/Desktop/dataset/en/validated_mels" 

os.makedirs(output_dir, exist_ok=True)

def normalize_melspectrogram(mel_spec):
    min_val = np.min(mel_spec)
    max_val = np.max(mel_spec)
    return (mel_spec - min_val) / (max_val - min_val)

def mel_spectrogram2d(audio_path):
    audio, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)  # Convert to decibel scale
    return normalize_melspectrogram(mel_spec_db)


for file in os.listdir(input_dir):
    if file.endswith(".mp3") or file.endswith(".wav"):  
        base_name = os.path.splitext(file)[0]  
        audio_path = os.path.join(input_dir, file)  
        text_path = os.path.join(input_dir, f"{base_name}.txt")  

        if os.path.exists(text_path):
          
            mel_spec = mel_spectrogram2d(audio_path)

            mel_file_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(mel_file_path, mel_spec)

            label_file_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(text_path, 'r', encoding='utf-8') as source_label, open(label_file_path, 'w', encoding='utf-8') as dest_label:
                dest_label.write(source_label.read())

            print(f"Processed: {file}")
        else:
            print(f"Text file not found for: {file}")

print(f"Mel spectrograms and labels saved to {output_dir}.")
