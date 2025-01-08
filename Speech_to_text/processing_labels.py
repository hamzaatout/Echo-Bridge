import numpy as np
import os

# Extended dictionary to map numbers to characters:
# - 'a' to 'z' -> 0-25
# - 'A' to 'Z' -> 26-51
# Special characters included
# Blank charavter is 60
id_to_char = {i: chr(97 + i) if i < 26 else (
    chr(65 + i - 26) if i < 52 else (
        ' ' if i == 52 else (
            '.' if i == 53 else (
                ',' if i == 54 else (
                    '?' if i == 55 else (
                        '!' if i == 56 else (
                            '/' if i == 57 else (
                                '\\' if i == 58 else (
                                    '(' if i == 59 else ')'))))))))) for i in range(60)}

char_to_id = {v: k for k, v in id_to_char.items()}

def transcription_to_ids(transcription):
    return np.array([char_to_id[char] for char in transcription if char in char_to_id])


def ids_to_transcription(transcription_ids):
    return ''.join([id_to_char[id] for id in transcription_ids])


input_dir = "/Users/karim/Desktop/dataset/en/validated_mels"  

for file in os.listdir(input_dir):
    if file.endswith(".txt"): 
        text_file_path = os.path.join(input_dir, file)  

        with open(text_file_path, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()

        transcription_ids = transcription_to_ids(transcription)

        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(map(str, transcription_ids)))

        print(f"Processed and updated: {file}")