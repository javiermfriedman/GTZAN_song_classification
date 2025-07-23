
"""
This script applies audio augmentation techniques to create additional training data for music genre classification.
It reads audio files from the source directory, applies augmentations, and saves both original and augmented files to a target directory.
"""

from dotenv import load_dotenv
import os
import numpy as np
import librosa
import soundfile as sf

load_dotenv()
source_dir = os.getenv("GENRES_ORIGINAL_PATH", "Data/genres_original")
target_dir = os.getenv("GENRES_AUGMENTED_PATH", "Data/genres_augmented")


def add_noise(y, noise_factor=0.005):
    """Adds random noise to an audio signal."""
    noise = np.random.randn(len(y))
    data_noise = y + noise_factor * noise
    return data_noise

def time_shift(y, sr, shift_max_sec=2):
    """Shifts the audio signal in time."""
    shift = np.random.randint(sr * shift_max_sec)
    if shift > 0:
        data_shift = np.pad(y, (shift, 0), mode='constant')[:len(y)]
    else:
        data_shift = np.pad(y, (0, -shift), mode='constant')[len(y):]
    return data_shift

def change_pitch_and_speed(y, sr, pitch_factor=0.7):
    """Changes the pitch and speed of the audio signal."""
    return librosa.effects.time_stretch(librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_factor), rate=pitch_factor)

def change_volume(y, gain_factor=0.3):
    """Changes the volume of the audio signal."""
    gain = np.random.uniform(1 - gain_factor, 1 + gain_factor)
    return y * gain

def pitch_shift(y, sr, n_steps_range=2):
    """Shifts the pitch without changing speed."""
    n_steps = np.random.uniform(-n_steps_range, n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

if __name__ == '__main__':
    """
    Augments audio data by creating variations of original files.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        exit()

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    for genre_folder in os.listdir(source_dir):
        source_genre_path = os.path.join(source_dir, genre_folder)
        target_genre_path = os.path.join(target_dir, genre_folder)
        
        if not os.path.isdir(source_genre_path):
            continue

        # Create genre directory in target
        if not os.path.exists(target_genre_path):
            os.makedirs(target_genre_path)

        print(f"Augmenting data for genre: {genre_folder}")

        for filename in os.listdir(source_genre_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(source_genre_path, filename)
                y, sr = librosa.load(file_path, sr=None)

                # Copy original file to target directory
                original_target_path = os.path.join(target_genre_path, filename)
                sf.write(original_target_path, y, sr)
                print(f"Copied original: {filename}")

                # 1. Add Noise
                y_noise = add_noise(y)
                sf.write(os.path.join(target_genre_path, f"{os.path.splitext(filename)[0]}_noise.wav"), y_noise, sr)

                # # 2. Time Shift
                # y_shifted = time_shift(y, sr)
                # sf.write(os.path.join(target_genre_path, f"{os.path.splitext(filename)[0]}_shifted.wav"), y_shifted, sr)

                # # 3. Change Pitch and Speed
                # y_pitched_sped = change_pitch_and_speed(y, sr)
                # sf.write(os.path.join(target_genre_path, f"{os.path.splitext(filename)[0]}_pitched_sped.wav"), y_pitched_sped, sr)

                # 4. Change Volume
                y_volume = change_volume(y)
                sf.write(os.path.join(target_genre_path, f"{os.path.splitext(filename)[0]}_volume.wav"), y_volume, sr)

                # 5. Pitch Shift (without speed change)
                y_pitch = pitch_shift(y, sr)
                sf.write(os.path.join(target_genre_path, f"{os.path.splitext(filename)[0]}_pitch.wav"), y_pitch, sr)

    print("Data augmentation complete.")


