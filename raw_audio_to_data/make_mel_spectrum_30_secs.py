#!/usr/bin/env python3
"""
This script converts audio files to mel spectrograms for music genre classification.
It processes audio files from a source directory and saves mel spectrograms as .npy files and .png images.
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
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        mel_spect = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=2048,      
            hop_length=512,   
            n_mels=128       
        )
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
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
    source_data_path = os.getenv("GENRES_AUGMENTED_PATH", "Data/genres_augmented")
    destination_path = os.getenv("MEL_SPECTROGRAM_DATA_PATH", "Data/mel_spectrogram_data")

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created directory: {destination_path}")

    genres = [g for g in os.listdir(source_data_path) if os.path.isdir(os.path.join(source_data_path, g))]

    for genre in genres:
        source_genre_path = os.path.join(source_data_path, genre)
        dest_genre_path = os.path.join(destination_path, genre)

        if not os.path.exists(dest_genre_path):
            os.makedirs(dest_genre_path)

        for filename in sorted(os.listdir(source_genre_path)):
            if filename.endswith('.wav'):
                audio_path = os.path.join(source_genre_path, filename)
                output_filename_base = os.path.splitext(filename)[0] # takes out .wav in filename
                output_npy_path = os.path.join(dest_genre_path, output_filename_base + '.npy')
                output_png_path = os.path.join(dest_genre_path, output_filename_base + '.png')
                
                print(f"Processing {audio_path}...")
                
                # Generate mel spectrogram
                mel_spect_db, _ = generate_mel_spectrogram(audio_path)
                
                if mel_spect_db is not None:
                    # Save as .npy file
                    save_mel_spectrogram_npy(mel_spect_db, output_npy_path)
                    
                    # Save as .png image
                    save_mel_spectrogram_image(mel_spect_db, output_png_path)
    print("Finished processing all audio files.")



if __name__ == '__main__':
    main() 