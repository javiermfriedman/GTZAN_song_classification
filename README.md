# GTZAN_song_classification

1 agument data works and puts it into directory
- add noise
- change volume
- pitch shift

make mel spectrum data this file works and puts it into directory



raw audio to digital data to process
- mel spectrum data not MFCC

    * Mel spectrum
        * Very effective for classifying instruments and genres
    * Chroma Features (Chroma STFT)
        * Captures harmonic content: chords, keys, pitch classes
        * Great for melody-rich genres
    * Spectral Centroid
        * The "center of mass" of the spectrum — tells you where the energy of a sound is concentrated.
            * High centroid → bright, sharp, trebly (e.g. cymbals)
            * Low centroid → dull, bass-heavy (e.g. tuba)
    * Spectral Roll-off
        * Tells you how much high-frequency content is present.
    * Zero-Crossing Rate
        * How often the waveform crosses the zero amplitude line — reflects the noisiness or percussiveness of a signal.
            * High rate → noisy signals (e.g. hi-hats, white noise)
            * Low rate → smooth signals (e.g. vocals, cello)

cnn 1 final results: 
for 10 epochs
Test Loss: 0.9173367023468018
Test Accuracy: 0.7055583596229553
for 15 epochs test 2
Test Loss: 0.8296502828598022
Test Accuracy: 0.7230846285820007

cnn 2 
for 40 epochs
Test Loss: 0.7112028002738953
Test Accuracy: 0.8197295665740967 