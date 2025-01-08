import numpy as np
import tensorflow as tf
import os

def load_padded_data(output_dir):
    spectrograms = []
    labels = []

    for filename in sorted(os.listdir(output_dir)):
        if filename.startswith("padded_melspec") and filename.endswith(".npy"):
            mel_path = os.path.join(output_dir, filename)
            spectrograms.append(np.load(mel_path))
        elif filename.startswith("padded_label") and filename.endswith(".txt"):
            label_path = os.path.join(output_dir, filename)
            with open(label_path, 'r') as f:
                labels.append(list(map(int, f.read().strip().split())))

    
    X = np.array(spectrograms)  
    y = np.array(labels)        
    return X, y


output_dir = "/Users/karim/Desktop/dataset/en/padded_dataset"
batch_size = 32


X_train, y_train = load_padded_data(output_dir)

print("X_train shape:", X_train.shape)  
print("y_train shape:", y_train.shape)  

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)  
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)    

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor))
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
