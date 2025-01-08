import numpy as np
import os

def pad_spectrogram(spectrogram, max_time_steps):
    current_time_steps, mel_bins = spectrogram.shape
    if current_time_steps < max_time_steps:
        padding = max_time_steps - current_time_steps
        padded_spectrogram = np.pad(
            spectrogram,
            ((0, padding), (0, 0)),  # Pad time axis only
            mode="constant",
            constant_values=0
        )
    else:
        padded_spectrogram = spectrogram
    return padded_spectrogram

def pad_text_sequence(sequence, max_length, blank_token):
    if len(sequence) < max_length:
        padding = max_length - len(sequence)
        padded_sequence = np.pad(
            sequence,
            (0, padding),  # Pad to the right
            mode="constant",
            constant_values=blank_token
        )
    else:
        padded_sequence = sequence
    return padded_sequence


n_mels = 128 
blank_token = 60  
input_dir = "/Users/karim/Desktop/dataset/en/validated_mels"
output_dir = "/Users/karim/Desktop/dataset/en/padded_dataset"


os.makedirs(output_dir, exist_ok=True)

all_files = sorted([f for f in os.listdir(input_dir)])
audio_files = [f for f in all_files if f.endswith('.npy')]
text_files = [f for f in all_files if f.endswith('.txt')]

base_names = {os.path.splitext(f)[0] for f in audio_files} & {os.path.splitext(f)[0] for f in text_files}

if not base_names:
    raise ValueError("No matching .npy and .txt files found in the input directory.")


max_time_steps = 0
max_label_length = 0

for base_name in sorted(base_names):
    mel_path = os.path.join(input_dir, base_name + ".npy")
    label_path = os.path.join(input_dir, base_name + ".txt")


    spectrogram = np.load(mel_path)
    spectrogram = spectrogram.T if spectrogram.shape[0] == n_mels else spectrogram  # Ensure shape is (time_steps, n_mels)
    max_time_steps = max(max_time_steps, spectrogram.shape[0])


    with open(label_path, 'r') as f:
        label = list(map(int, f.read().strip().split()))
    max_label_length = max(max_label_length, len(label))

print(f"Max time steps: {max_time_steps}")
print(f"Max label length: {max_label_length}")


for idx, base_name in enumerate(sorted(base_names), start=1):
    mel_path = os.path.join(input_dir, base_name + ".npy")
    label_path = os.path.join(input_dir, base_name + ".txt")

  
    spectrogram = np.load(mel_path)
    spectrogram = spectrogram.T if spectrogram.shape[0] == n_mels else spectrogram  # Ensure shape is (time_steps, n_mels)
    padded_spectrogram = pad_spectrogram(spectrogram, max_time_steps)

    with open(label_path, 'r') as f:
        label = list(map(int, f.read().strip().split()))
    padded_label = pad_text_sequence(label, max_label_length, blank_token)

 
    mel_filename = f"padded_melspec_{idx}.npy"
    mel_path = os.path.join(output_dir, mel_filename)
    np.save(mel_path, padded_spectrogram)


    label_filename = f"padded_label_{idx}.txt"
    label_path = os.path.join(output_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write(" ".join(map(str, padded_label)))

print(f"Padded files saved to: {output_dir}")
