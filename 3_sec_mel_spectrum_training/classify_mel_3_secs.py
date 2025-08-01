"""
GTZAN Music Genre Classification - Main Training Script
======================================================

This script is the main entry point for training CNN models on the GTZAN music
genre dataset. It loads preprocessed mel spectrogram data, splits it into
training and validation sets, builds and compiles a CNN model, and trains it
for music genre classification.

Author: [Your Name]
Date: [Date]
Version: 1.0

Dependencies:
    - tensorflow.keras: For model building and training
    - sklearn.model_selection: For data splitting
    - numpy: For array operations
    - load_3_sec_mel_data: Custom module for data loading
    - model: Custom module for CNN architectures
    - model_utils: Custom module for visualization

Model Configuration:
    - Architecture: CNN Model 2 (with batch normalization and dropout)
    - Optimizer: Adam with learning rate 0.0001
    - Loss: Sparse categorical crossentropy
    - Metrics: Accuracy
    - Batch size: 64
    - Epochs: 40

Data Processing:
    - Input: 3-second mel spectrograms
    - Output: 10 genre classes
    - Train/Validation split: 80/20

Usage:
    python classify_mel_3_secs.py
    
Output:
    - Trained model saved to: models/cnn2_model_3.keras
    - Training history plot saved to: plots/cnn2_history_3.png
    - Console output with training progress and final metrics
"""

from load_3_sec_mel_data import get_3_sec_mel_data
from model import build_cnn_model_1, build_cnn_model_2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from model_utils import plot_history

X_data, Y_data = get_3_sec_mel_data()

print("X_data.shape: ", X_data.shape)
print("Y_data.shape: ", Y_data.shape)

# split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=20)
print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Add additional dimension for CNN
X_train = X_train[..., np.newaxis] # Add additional dimension for CNN
X_val = X_val[..., np.newaxis]  # Add additional dimension for CNN

print("X_train.shape: ", X_train.shape)
input_shape = X_train.shape[1:4]
# X_train_cnn.shape # shape = (# samples, time-bins (x), num MFCCs (y), "channel" (like an image))
model = build_cnn_model_2(input_shape)

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001), # can also use 'adam'
    loss='sparse_categorical_crossentropy', # loss for multi-class classification
    metrics=['acc']
)

# Train the model
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=40,
    verbose=1
)
# save the model
save_model(model, "3_sec_mel_spectrum_training/models/cnn2_model_3.keras")

loss_cnn, acc_cnn = model.evaluate(X_val, y_val)
print(f"Test Loss: {loss_cnn}")
print(f"Test Accuracy: {acc_cnn}")
plot_history(hist, "3_sec_mel_spectrum_training/plots/cnn2_history_3.png")







