import librosa
import os
import math
import json
import numpy as np
import soundfile as sf # Import soundfile

CLIP_DURATION = 3 # in seconds

if __name__ == '__main__':
    """
    slices 30 second audio files into 3 second clips
    """
    source_dir = os.getenv("GENRES_ORIGINAL_PATH", "Data/genres_original")
    target_dir = os.getenv("GENRES_3_SECONDS_AUGMENTED_PATH", "Data/genres_3_seconds_augmented")


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
                
                # create a new directroy for this 30 second file and save the 3 second clips in it
                sound_file_name = os.path.splitext(filename)[0]
                sound_file_dir = os.path.join(target_genre_path, sound_file_name)
                os.makedirs(sound_file_dir, exist_ok=True)

                # slice the 30 second file into 3 second clips
                try:
                    # Load the audio file
                    file_path = os.path.join(source_genre_path, filename)
                    audio, sr = librosa.load(file_path, sr=None) # sr=None preserves original sample rate

                    # Calculate total samples for the desired clip duration
                    samples_per_clip = sr * CLIP_DURATION

                    # Calculate the total number of full clips that can be created
                    num_clips = math.floor(len(audio) / samples_per_clip)

                    if num_clips == 0:
                        print(f"Warning: Audio file {file_path} is shorter than {CLIP_DURATION} seconds and cannot be split.")
                        continue

                    print(f"Splitting {file_path} into {num_clips} clips...")

                    # 3. Loop through the audio and create clips
                    for i in range(num_clips):
                        # Calculate start and end samples for each clip
                        start_sample = i * samples_per_clip
                        end_sample = start_sample + samples_per_clip
                        
                        # Get the clip's audio data
                        clip_data = audio[start_sample:end_sample]

                        # Create a unique filename for each clip
                        base_filename = os.path.splitext(os.path.basename(file_path))[0]
                        clip_filename = f"{base_filename}_clip_{i+1}.wav"
                        output_path = os.path.join(sound_file_dir, clip_filename)

                        # Save the clip to the output directory
                        sf.write(output_path, clip_data, sr)
                    
                    print(f"Successfully saved {num_clips} clips to {sound_file_dir}.")

                except Exception as e:
                    # Catch any other errors during file processing (e.g., corrupt file)
                    print(f"An error occurred while processing {filename}: {e}")


               

    print("Data augmentation complete.")
