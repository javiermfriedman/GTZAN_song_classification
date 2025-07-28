"""
GTZAN Music Genre Classification - CNN Model Architectures
========================================================

This module defines two Convolutional Neural Network (CNN) architectures designed
for music genre classification using mel spectrograms. The models are specifically
tailored for processing 3-second audio clips converted to mel spectrogram format.

Author: Javier Friedman

Dependencies:
    - tensorflow.keras: For building neural network models
    - tensorflow.keras.layers: For layer definitions
    - tensorflow.keras.models: For Sequential model

Model Architectures:
    1. build_cnn_model_1: Simple CNN with basic convolutional blocks
    2. build_cnn_model_2: Advanced CNN with batch normalization and dropout

Input Shape:
    - Expected input: (time_bins, mel_bins, 1) - 3D tensor representing mel spectrogram
    
Output:
    - 10 classes corresponding to music genres: blues, classical, country, disco,
      hiphop, jazz, metal, pop, reggae, rock

Usage:
    from model import build_cnn_model_1, build_cnn_model_2
    
    # For simple model
    model = build_cnn_model_1(input_shape)
    
    # For advanced model with regularization
    model = build_cnn_model_2(input_shape)
"""

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

# this model will be a simple CNN model, for processing numpy data of mel for 3 second clips
def build_cnn_model_1(input_shape):
    model_cnn1 = Sequential()

    # Create a convolution block
    model_cnn1.add(Conv2D(32, 3, activation='relu', input_shape=input_shape)) # first hidden conv layer
    # -- Conv2D: Extracts 32 feature maps using 3x3 filters from the input mel spectrogram
    model_cnn1.add(MaxPooling2D(3, strides=(2,2), padding='same')) # MaxPool the results
    # -- MaxPooling2D: Reduces the spatial dimensions, keeping the most important features and reducing computation

    # Add another conv block
    model_cnn1.add(Conv2D(64, 3, activation='relu'))
    model_cnn1.add(MaxPooling2D(3, strides=(2,2), padding='same'))

    # Add another conv block
    model_cnn1.add(Conv2D(64, 2, activation='relu'))
    model_cnn1.add(MaxPooling2D(2, strides=(2,2), padding='same'))

    # Flatten output to send through dense layers
    model_cnn1.add(Flatten())
    model_cnn1.add(Dense(64, activation='relu'))

    # output to 10 classes for predictions
    model_cnn1.add(Dense(10, activation='softmax'))

    return model_cnn1

def build_cnn_model_2(input_shape):
    model_cnn2 = Sequential()

    # Create a convolution block
    model_cnn2.add(Conv2D(32, 3, activation='relu', input_shape=input_shape)) # first hidden conv layer
    model_cnn2.add(BatchNormalization())
    model_cnn2.add(MaxPooling2D(3, strides=(2,2), padding='same')) # MaxPool the results
    model_cnn2.add(Dropout(0.2))

    # Add another conv block
    model_cnn2.add(Conv2D(64, 3, activation='relu'))
    model_cnn2.add(BatchNormalization())
    model_cnn2.add(MaxPooling2D(3, strides=(2,2), padding='same'))
    model_cnn2.add(Dropout(0.1))

    # Add another conv block
    model_cnn2.add(Conv2D(64, 2, activation='relu'))
    model_cnn2.add(BatchNormalization())
    model_cnn2.add(MaxPooling2D(2, strides=(2,2), padding='same'))
    model_cnn2.add(Dropout(0.1))

    # Flatten output to send through dense layers
    model_cnn2.add(Flatten())
    model_cnn2.add(Dense(128, activation='relu'))
    model_cnn2.add(Dropout(0.5))

    # output to 10 classes for predictions
    model_cnn2.add(Dense(10, activation='softmax')) # Softmax activation for multi-class classification

    return model_cnn2