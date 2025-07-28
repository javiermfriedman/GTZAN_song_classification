"""
GTZAN Music Genre Classification - Model Testing and Inference Script
==================================================================

This script provides an interactive interface for testing trained CNN models on
the GTZAN dataset. It allows users to select specific tracks from the dataset,
process them into 3-second segments, and obtain genre predictions for each
segment. The final prediction is determined by aggregating probabilities across
all segments of a track.

Author: Javier Friedman

Dependencies:
    - tensorflow.keras: For model loading and prediction
    - librosa: For audio loading and processing
    - numpy: For numerical operations
    - math: For mathematical operations
    - os: For file system operations
    - dotenv: For environment variable management
    - inference_utils: Custom module for audio processing and user input

Model Requirements:
    - Trained CNN model saved as: models/cnn2_model_3.keras
    - Model expects input shape: (1, time_bins, mel_bins, 1)

Data Processing:
    - Audio tracks are divided into 3-second segments
    - Each segment is converted to mel spectrogram
    - Predictions are aggregated across all segments
    - Final genre prediction is based on highest probability

Supported Genres:
    1. blues      2. classical   3. country     4. disco
    5. hiphop     6. jazz        7. metal       8. pop
    9. reggae    10. rock

Environment Variables:
    - GENRES_ORIGINAL_PATH: Path to original genre audio files (default: Data/genres_original)

Usage:
    python test_model.py
    
    Follow the interactive prompts to:
    1. Select a genre (1-10)
    2. Enter track number (00000-00099)
    3. View predictions for each 3-second segment
    4. See final aggregated genre prediction
"""

from model import build_cnn_model_1, build_cnn_model_2
from load_3_sec_mel_data import get_3_sec_mel_data
from tensorflow.keras.models import load_model
import librosa
from inference_utils import prompt_input
import numpy as np
import math
from inference_utils import generate_mel_spectrogram
from dotenv import load_dotenv
import os

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

def main():

# load the model
    model = load_model("3_sec_mel_spectrum_training/models/cnn2_model_3.keras")

    while True:

        # 1. get the audio track for analysis
        genre, track = prompt_input()
        if genre is None:
            break


        base_path = os.getenv("GENRES_ORIGINAL_PATH", "Data/genres_original")
        file_path = f"{base_path}/{genre}/{genre}.{track}.wav"
        try:
            print(f"trying to get track from path:  {file_path}")
            audio, sample_rate = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error invalid track: {file_path}: {e}")
            continue

        # 2. Initialize a sum vector for probabilities of each genre
        total_probs = np.zeros(model.output_shape[1])

        try:
            # 3. how many 3 second clips are in the track
            samples_per_clip = sample_rate * 3
            num_clips = math.floor(len(audio) / samples_per_clip)
           
            for i in range(num_clips):
                # 4. cut out the 3 second clip for anlaysis 
                start_sample = i * samples_per_clip
                end_sample = start_sample + samples_per_clip
                clip_data = audio[start_sample:end_sample]

                # 5. turn into mel spectrogram to pass into model
                mel_spectrogram_np, sample_rate = generate_mel_spectrogram(clip_data, sample_rate)

                mel_spectrogram_np = mel_spectrogram_np[np.newaxis, ..., np.newaxis]

                # 6. pass into model
                predictions = model.predict(mel_spectrogram_np)
                print(f"prediction for clip {i}: {predictions}")

                # 7. add the predictions to the total probabilities
                total_probs += predictions[0]
                            
        except Exception as e:
            # Catch any other errors during file processing (e.g., corrupt file)
            print(f"An error occurred while processing {file_path}: {e}")

        # 9. Print the total and average probabilities
        print("Total probabilities for each genre:", total_probs)
        INDEX_TO_GENRE = {v: k for k, v in GENRE_TO_INDEX.items()}

        predicted_index = np.argmax(total_probs)
        predicted_genre = INDEX_TO_GENRE[predicted_index]
        print("Most likely genre:", predicted_genre)

        



# test the model    
if __name__ == "__main__":
    main()



