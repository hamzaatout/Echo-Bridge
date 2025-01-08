import numpy as np
import tensorflow as tf
import os


def check_invalid_lengths_in_directory(output_dir):
    """
    Loops through all spectrogram and label files in the directory,
    checks for invalid lengths, and prints warnings if any file does
    not match the reference shapes.
    """
    spectrograms = []
    labels = []

    for filename in sorted(os.listdir(output_dir)):
        if filename.startswith("padded_melspec") and filename.endswith(".npy"):
            if len(spectrograms) == 0:
                reference_spectrogram = np.load(os.path.join(output_dir, filename))
                reference_spectrogram_shape = reference_spectrogram.shape
                print(f"Reference spectrogram shape: {reference_spectrogram_shape}")
            spectrograms.append(filename)
        elif filename.startswith("padded_label") and filename.endswith(".txt"):
            if len(labels) == 0:
                with open(os.path.join(output_dir, filename), 'r') as f:
                    reference_label = list(map(int, f.read().strip().split()))
                    reference_label_length = len(reference_label)
                    print(f"Reference label length: {reference_label_length}")
            labels.append(filename)

    for idx, filename in enumerate(sorted(spectrograms), start=1):
        filepath = os.path.join(output_dir, filename)
        spectrogram = np.load(filepath)
        if spectrogram.shape != reference_spectrogram_shape:
            print(f"Invalid spectrogram shape at index {idx} ({filename}): {spectrogram.shape} (expected {reference_spectrogram_shape})")

    for idx, filename in enumerate(sorted(labels), start=1):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r') as f:
            label = list(map(int, f.read().strip().split()))
            if len(label) != reference_label_length:
                print(f"Invalid label length at index {idx} ({filename}): {len(label)} (expected {reference_label_length})")

output_dir = "/Users/karim/Desktop/dataset/en/padded_dataset"

check_invalid_lengths_in_directory(output_dir)
