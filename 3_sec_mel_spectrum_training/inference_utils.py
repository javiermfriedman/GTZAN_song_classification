import os
import librosa
import numpy as np



def generate_mel_spectrogram(audio_clip, sample_rate):
    """
    Generates a mel spectrogram from an audio clip of 3 second.
    Returns the mel spectrogram in dB in numpy array.
    """
    try:
        mel_spect = librosa.feature.melspectrogram(
            y=audio_clip,              # Audio time series
            sr=sample_rate,            # Sampling rate
            n_fft=2048,       # Length of the FFT window
            hop_length=512,   # Number of samples between successive frames
            n_mels=128        # Number of Mel bands to generate
        )
        # Convert the mel spectrogram to decibel (dB) units
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        # Return the mel spectrogram in dB and the sample rate
        return mel_spect_db, sample_rate
    except Exception as e:
        print(f"Error processing {audio_clip}: {e}")
        return None, None


def prompt_input():
    print("1. blues")
    print("2. classical")
    print("3. country")
    print("4. disco")
    print("5. hiphop")
    print("6. jazz")
    print("7. metal")
    print("8. pop")
    print("9. reggae")
    print("10. rock")
    print("q. quit")
    
    input_genre = input("which genre: ")

    genre = ""
    if input_genre == "1":
        genre = "blues"
    elif input_genre == "2":
        genre = "classical"
    elif input_genre == "3":
        genre = "country"
    elif input_genre == "4":
        genre = "disco"
    elif input_genre == "5":
        genre = "hip-hop"
    elif input_genre == "6":
        genre = "jazz"
    elif input_genre == "7":
        genre = "metal"
    elif input_genre == "8":
        genre = "pop"
    elif input_genre == "9":
        genre = "reggae"
    elif input_genre == "10":
        genre = "rock"
    elif input_genre == "q":
        return None, None
    else:
        return None, None

    input_track = input("selected track format 00000-00099:")

    return genre, input_track