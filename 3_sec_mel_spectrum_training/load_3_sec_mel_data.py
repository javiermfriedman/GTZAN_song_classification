"""
GTZAN Music Genre Classification - Data Loading Module
====================================================

This module provides functionality to load and prepare the GTZAN music genre dataset
for training deep learning models. It handles the loading of 3-second mel spectrogram
data that has been preprocessed and stored as .npy files.

Author: [Your Name]
Date: [Date]
Version: 1.0

Dependencies:
    - numpy: For array operations
    - os: For file system operations
    - dotenv: For environment variable management

Environment Variables:
    - MEL_DATA_PATH: Path to the mel spectrogram data directory (default: Data/mel_spectrogram_data_3_seconds)

Usage:
    from load_3_sec_mel_data import get_3_sec_mel_data
    
    X_data, Y_data = get_3_sec_mel_data()
    print(f"Data shape: {X_data.shape}")
    print(f"Labels shape: {Y_data.shape}")

Returns:
    - X_data: numpy array of mel spectrograms with shape (n_samples, time_bins, mel_bins)
    - Y_data: numpy array of genre labels (0-9 corresponding to 10 genres)
"""

import os
import numpy as np
from dotenv import load_dotenv

GENRE_TO_INDEX = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}

load_dotenv()

def get_3_sec_mel_data():
    """
    Loads the GTZAN 3 second mel spectrum dataset from .npy
    """

    load_dotenv()
    source_data_path = os.getenv("ROOT_DIR_PATH", "Data/mel_spectrogram_data_3_seconds")
    if not os.path.isdir(source_data_path):
        raise FileNotFoundError(f"[Error] The directory {source_data_path} was not found. Please ensure your data is processed and in the correct location.")

    print(f"[INFO] Loading mel data from: {source_data_path}")

    X_data = []
    Y_data = []

    

    for genre in os.listdir(source_data_path):
        genre_path = os.path.join(source_data_path, genre)
        if not os.path.isdir(genre_path): 
            continue

        
        y_label = GENRE_TO_INDEX.get(genre)
        # print("y_label: ", y_label)
        # print("genre: ", genre)

        for filename in sorted(os.listdir(genre_path)):
            if filename.endswith(".npy"):
                # print("filename: ", filename)
                file_path = os.path.join(genre_path, filename)
                
                np_data = np.load(file_path)
                # print("np_data.shape: ", np_data.shape)
                X_data.append(np_data)
                Y_data.append(y_label)



    # Convert lists to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    return X_data, Y_data

if __name__ == "__main__":
    X_data, Y_data = get_3_sec_mel_data()
    print("X_data.shape: ", X_data.shape)
    print("Y_data.shape: ", Y_data.shape)


