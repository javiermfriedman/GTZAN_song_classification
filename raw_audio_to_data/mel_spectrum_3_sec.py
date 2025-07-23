#!/usr/bin/env python3
"""
this script takes 3 second audio files and converts them to mel spectrograms
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

def generate_mel_spectrogram(audio_path):
    """
    Generates a mel spectrogram from an audio file.
    Returns the mel spectrogram in dB and the sample rate.
    """
    try:
        # Load the audio file at the given path, resample to 22050 Hz, convert to mono
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        # Compute the mel spectrogram from the audio signal
        mel_spect = librosa.feature.melspectrogram(
            y=y,              # Audio time series
            sr=sr,            # Sampling rate
            n_fft=2048,       # Length of the FFT window
            hop_length=512,   # Number of samples between successive frames
            n_mels=128        # Number of Mel bands to generate
        )
        # Convert the mel spectrogram to decibel (dB) units
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        # Return the mel spectrogram in dB and the sample rate
        return mel_spect_db, sr
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def save_mel_spectrogram_npy(mel_spect_db, output_path):
    """
    Saves a mel spectrogram as a .npy file.
    """
    np.save(output_path, mel_spect_db)
    print(f"Saved mel spectrogram as {output_path}")

def save_mel_spectrogram_image(mel_spect_db, output_path):
    """
    Saves a mel spectrogram as a PNG image without any axes or color bars.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect_db, sr=22050, hop_length=512)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved mel spectrogram image as {output_path}")



def main():
    source_data_path = os.getenv("GENRES_3_SECONDS_AUGMENTED_PATH", "Data/genres_3_seconds_augmented")
    destination_path = os.getenv("MEL_SPECTROGRAM_DATA_3_SECONDS_PATH", "Data/mel_spectrogram_data_3_seconds")

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created directory: {destination_path}")

    genres = [g for g in os.listdir(source_data_path) if os.path.isdir(os.path.join(source_data_path, g))]

    for genre in genres:
        source_genre_path = os.path.join(source_data_path, genre)
        destination_genre_path = os.path.join(destination_path, genre)

        if not os.path.exists(destination_genre_path):
            os.makedirs(destination_genre_path)

        for song_dir in sorted(os.listdir(source_genre_path)):
            song_dir_path = os.path.join(source_genre_path, song_dir)
            if not os.path.isdir(song_dir_path):
                continue  # Skip files like .DS_Store
            for filename in os.listdir(song_dir_path):

                if filename.endswith('.wav'):
                    audio_path = os.path.join(song_dir_path, filename)
                    output_filename_base = os.path.splitext(filename)[0] # takes out .wav in filename
                    output_npy_path = os.path.join(destination_genre_path, output_filename_base + '.npy')
                    # output_png_path = os.path.join(destination_genre_path, output_filename_base + '.png')
                    
                    print(f"Processing {audio_path}...")
                    
                    # Generate mel spectrogram
                    mel_spect_db, _ = generate_mel_spectrogram(audio_path)
                    
                    if mel_spect_db is not None:
                        # Save as .npy file
                        save_mel_spectrogram_npy(mel_spect_db, output_npy_path)
                        # # Save as .png image
                        # save_mel_spectrogram_image(mel_spect_db, output_png_path)
    print("Finished processing all audio files.")



if __name__ == '__main__':
    main() 