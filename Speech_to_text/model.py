import tensorflow as tf
import numpy as np
import os
from data_loading import train_dataset, X_train, y_train, X_train_tensor, y_train_tensor

n_mels = 128  
vocab_size = 61  
batch_size = 250
data_dir = "/Users/karim/Desktop/dataset/en/validated_mels"
blank_token = 60  #ID for the blank token in CTC

#LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),  
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
    tf.keras.layers.Dense(vocab_size + 1, activation='softmax') 
])
'''
#Vanilla RNN model
model = tf.keras.Sequential([
    # Input shape: (time_steps, mel_bins)
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # Replace LSTM with SimpleRNN
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(512, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(512, return_sequences=True)),
    
    # TimeDistributed dense layer
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
    
    # Final dense layer with softmax activation
    tf.keras.layers.Dense(vocab_size + 1, activation='softmax')  
])
'''

model.summary()

def ctc_loss(y_true, y_pred):
    
    y_pred = tf.transpose(y_pred, perm=[1, 0, 2])
    y_true = tf.cast(y_true, tf.int32)
    label_length = tf.reduce_sum(tf.cast(y_true != blank_token, tf.int32), axis=1)
    input_length = tf.fill([tf.shape(y_true)[0]], tf.shape(y_pred)[0])

    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        blank_index=blank_token
    )
    return tf.reduce_mean(loss) 



model.compile(optimizer='adam', loss=ctc_loss)

history = model.fit(train_dataset, epochs=200)

