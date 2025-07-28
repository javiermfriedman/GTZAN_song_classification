"""
GTZAN Music Genre Classification - Full Dataset Testing Script
============================================================

This script automatically tests the trained CNN model on the entire GTZAN dataset,
processing every song across all genres. It provides comprehensive analysis including
accuracy metrics, confusion matrix, per-genre performance, and detailed reporting.

Author: Javier Friedman

Dependencies:
    - tensorflow.keras: For model loading and prediction
    - librosa: For audio loading and processing
    - numpy: For numerical operations
    - pandas: For data analysis and reporting
    - matplotlib: For visualization
    - seaborn: For enhanced plotting
    - sklearn.metrics: For performance metrics
    - os: For file system operations
    - dotenv: For environment variable management
    - inference_utils: Custom module for audio processing

Model Requirements:
    - Trained CNN model saved as: models/cnn2_model_3.keras
    - Model expects input shape: (1, time_bins, mel_bins, 1)

Output:
    - Comprehensive accuracy report
    - Confusion matrix visualization
    - Per-genre performance analysis
    - Detailed CSV report with all predictions
    - Summary statistics

Usage:
    python test_full_dataset.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
import librosa
import math
from inference_utils import generate_mel_spectrogram
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Genre mappings
GENRE_TO_INDEX = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
    'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
}

INDEX_TO_GENRE = {v: k for k, v in GENRE_TO_INDEX.items()}

class DatasetTester:
    """
    Comprehensive dataset testing class for GTZAN music genre classification.
    """
    
    def __init__(self, model_path="3_sec_mel_spectrum_training/models/cnn2_model_3.keras"):
        """
        Initialize the dataset tester.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = load_model(model_path)
        self.base_path = os.getenv("GENRES_ORIGINAL_PATH", "Data/genres_original")
        self.results = []
        self.true_labels = []
        self.predicted_labels = []
        
    def process_single_track(self, genre, track_num):
        """
        Process a single track and return predictions.
        
        Args:
            genre (str): Genre name
            track_num (str): Track number (e.g., "00000")
            
        Returns:
            dict: Prediction results for the track
        """
        file_path = f"{self.base_path}/{genre}/{genre}.{track_num}.wav"
        
        try:
            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Calculate number of 3-second clips
            samples_per_clip = sample_rate * 3
            num_clips = math.floor(len(audio) / samples_per_clip)
            
            if num_clips == 0:
                return None
            
            # Initialize probability accumulator
            total_probs = np.zeros(self.model.output_shape[1])
            clip_predictions = []
            
            # Process each 3-second clip
            for i in range(num_clips):
                start_sample = i * samples_per_clip
                end_sample = start_sample + samples_per_clip
                clip_data = audio[start_sample:end_sample]
                
                # Generate mel spectrogram
                mel_spectrogram_np, _ = generate_mel_spectrogram(clip_data, sample_rate)
                if mel_spectrogram_np is None:
                    continue
                
                # Prepare for model input
                mel_spectrogram_np = mel_spectrogram_np[np.newaxis, ..., np.newaxis]
                
                # Get prediction
                predictions = self.model.predict(mel_spectrogram_np, verbose=0)
                clip_predictions.append(predictions[0])
                total_probs += predictions[0]
            
            # Calculate final prediction
            predicted_index = np.argmax(total_probs)
            predicted_genre = INDEX_TO_GENRE[predicted_index]
            true_genre_index = GENRE_TO_INDEX[genre]
            
            # Calculate confidence (max probability)
            confidence = np.max(total_probs) / num_clips
            
            return {
                'genre': genre,
                'track': track_num,
                'file_path': file_path,
                'true_label': genre,
                'predicted_label': predicted_genre,
                'true_index': true_genre_index,
                'predicted_index': predicted_index,
                'confidence': confidence,
                'num_clips': num_clips,
                'correct': genre == predicted_genre,
                'total_probabilities': total_probs.tolist(),
                'clip_predictions': [pred.tolist() for pred in clip_predictions]
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def test_all_tracks(self):
        """
        Test all tracks in the dataset.
        """
        print("Starting full dataset testing...")
        print(f"Base path: {self.base_path}")
        
        total_tracks = 0
        processed_tracks = 0
        start_time = time.time()
        
        # Iterate through all genres
        for genre in GENRE_TO_INDEX.keys():
            genre_path = os.path.join(self.base_path, genre)
            if not os.path.exists(genre_path):
                print(f"Warning: Genre path {genre_path} does not exist")
                continue
                
            print(f"\nProcessing genre: {genre}")
            
            # Get all wav files in the genre directory
            wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            wav_files.sort()
            
            for wav_file in wav_files:
                total_tracks += 1
                
                # Extract track number from filename (e.g., "blues.00000.wav" -> "00000")
                track_num = wav_file.split('.')[1]
                
                # Process the track
                result = self.process_single_track(genre, track_num)
                
                if result is not None:
                    self.results.append(result)
                    self.true_labels.append(result['true_index'])
                    self.predicted_labels.append(result['predicted_index'])
                    processed_tracks += 1
                    
                    # Print progress
                    if processed_tracks % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {processed_tracks} tracks in {elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        print(f"\nTesting completed!")
        print(f"Total tracks found: {total_tracks}")
        print(f"Successfully processed: {processed_tracks}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Average time per track: {elapsed_time/processed_tracks:.2f} seconds")
    
    def generate_report(self):
        """
        Generate comprehensive performance report.
        """
        if not self.results:
            print("No results to analyze!")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        
        # Calculate per-genre accuracy
        genre_accuracy = {}
        for genre in GENRE_TO_INDEX.keys():
            genre_df = df[df['genre'] == genre]
            if len(genre_df) > 0:
                genre_accuracy[genre] = genre_df['correct'].mean()
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET TESTING RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Total Tracks Tested: {len(self.results)}")
        
        print("\nPer-Genre Accuracy:")
        for genre, acc in genre_accuracy.items():
            count = len(df[df['genre'] == genre])
            print(f"  {genre:10}: {acc:.4f} ({acc*100:.2f}%) - {count} tracks")
        
        # Generate confusion matrix
        self.plot_confusion_matrix()
        
        # Save detailed results
        self.save_detailed_results(df)
        
        return df
    
    def plot_confusion_matrix(self):
        """
        Plot and save confusion matrix.
        """
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(GENRE_TO_INDEX.keys()),
                   yticklabels=list(GENRE_TO_INDEX.keys()))
        plt.title('Confusion Matrix - GTZAN Dataset Testing')
        plt.xlabel('Predicted Genre')
        plt.ylabel('True Genre')
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"3_sec_mel_spectrum_training/plots/confusion_matrix_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {plot_path}")
        plt.show()
    
    def save_detailed_results(self, df):
        """
        Save detailed results to CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"3_sec_mel_spectrum_training/results/detailed_results_{timestamp}.csv"
        
        # Create results directory if it doesn't exist
        os.makedirs("3_sec_mel_spectrum_training/results", exist_ok=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # Create summary statistics
        summary_stats = {
            'overall_accuracy': accuracy_score(self.true_labels, self.predicted_labels),
            'total_tracks': len(df),
            'correct_predictions': df['correct'].sum(),
            'average_confidence': df['confidence'].mean(),
            'average_clips_per_track': df['num_clips'].mean()
        }
        
        # Add per-genre statistics
        for genre in GENRE_TO_INDEX.keys():
            genre_df = df[df['genre'] == genre]
            if len(genre_df) > 0:
                summary_stats[f'{genre}_accuracy'] = genre_df['correct'].mean()
                summary_stats[f'{genre}_count'] = len(genre_df)
        
        # Save summary
        summary_path = f"3_sec_mel_spectrum_training/results/summary_{timestamp}.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
        print(f"Summary statistics saved to: {summary_path}")

def main():
    """
    Main function to run the full dataset testing.
    """
    print("GTZAN Full Dataset Testing")
    print("="*40)
    
    # Initialize tester
    tester = DatasetTester()
    
    # Test all tracks
    tester.test_all_tracks()
    
    # Generate and display report
    results_df = tester.generate_report()
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    main() 