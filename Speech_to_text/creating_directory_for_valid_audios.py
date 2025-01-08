import tensorflow as tf
import librosa
import pandas as pd
import numpy as np
import librosa.display
import string
import pandas as pd
import os
import shutil

metadata_path = "/Users/karim/Desktop/dataset/en/validated.tsv" 
clips_dir = "/Users/karim/Desktop/dataset/en/clips" 
output_dir = "/Users/karim/Desktop/dataset/en/validated_clips"  

os.makedirs(output_dir, exist_ok=True)

print("Loading validated metadata...")
validated_metadata = pd.read_csv(metadata_path, sep='\t')

for index, row in validated_metadata.iterrows():
    audio_file = row['path'] 
    label = row['sentence'] 
    
    source_path = os.path.join(clips_dir, audio_file)

    new_base_name = f"audio{index + 1}" 
    destination_audio_path = os.path.join(output_dir, f"{new_base_name}.mp3")
    destination_text_path = os.path.join(output_dir, f"{new_base_name}.txt")

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_audio_path)
        with open(destination_text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(label)
    else:
        print(f"Audio file not found: {source_path}")

print(f"Processed {len(validated_metadata)} validated audio files and their labels.")
