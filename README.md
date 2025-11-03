# Hear Me Out: Emotion Recognition from Audio

## Project Overview

This project trains deep learning models to detect emotions (Happy, Sad, Angry, Neutral) from voice clips using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. We compare feature extraction methods (MFCCs vs Mel-spectrograms) and test model robustness to audio noise.

## Dataset Structure

The RAVDESS dataset is organized in the following folder structure:

```
data/
├── Audio_Song_Actors_01-24_Actors_1_to_17/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── ...
│   └── Actor_17/
├── Audio_Song_Actors_01-24_Actors_19_to_24/
│   ├── Actor_19/
│   ├── Actor_20/
│   ├── ...
│   └── Actor_24/
├── Audio_Speech_Actors_01-24_Actors_1_to_17/
│   └── [Similar structure]
├── Audio_Speech_Actors_01-24_Cut_Actors_19_to_24/
│   └── [Similar structure]
└── Video_* folders
    └── [Video data, not used for audio-only models]
```

### Filename Encoding

RAVDESS filenames encode metadata in this format:
```
Modality-Vocal Channel-Emotion-Intensity-Statement-Repetition-Actor.wav

Example: 02-01-04-01-01-01-01.wav
- 02 = Song
- 01 = Female (01=female, 02=male)
- 04 = Angry emotion
- 01 = Normal intensity
- 01 = Statement 1
- 01 = First repetition
- 01 = Actor 01
```

**Emotion Mapping:**
- 01 = Neutral
- 02 = Happy
- 03 = Sad
- 04 = Angry

## Features Extracted

### MFCCs (Mel-Frequency Cepstral Coefficients)
- 13 coefficients extracted per audio file
- Captures perceptual loudness characteristics
- Averaged across time axis for fixed-size input
- Good for speech analysis

### Mel-Spectrograms
- 128 mel-frequency bins
- Time-frequency representation
- Averaged across time axis
- Captures spectral content

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or with conda:
   ```bash
   conda create -n hear_me_out python=3.10
   conda activate hear_me_out
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python -c "import librosa; import tensorflow; print('All dependencies installed!')"
   ```

## Running the Pipeline

### Quick Start - Jupyter Notebooks

1. **Data Exploration**
   ```bash
   jupyter notebook ravdess_starter.ipynb
   ```
   This notebook demonstrates:
   - Loading audio files
   - Visualizing waveforms and spectrograms
   - Extracting MFCC features

2. **Full ML Pipeline**
   ```bash
   jupyter notebook emotion_recognition_full_pipeline.ipynb
   ```
   This notebook includes:
   - Feature extraction (MFCC & Mel-spectrogram)
   - Model training (Baseline CNN & CNN-LSTM)
   - Model evaluation and comparison

### Production Training Script

Run the complete pipeline from command line:
```bash
python train_emotion_recognition.py
```

This will:
- Load all audio files from the data directories
- Extract features from all samples
- Train both model architectures
- Generate evaluation metrics and plots
- Save trained models to disk

### Configuration

Edit `train_emotion_recognition.py` to modify:
- `sample_rate`: Audio sampling rate (default: 22050 Hz)
- `n_mels`: Number of mel-frequency bins (default: 128)
- `n_mfcc`: Number of MFCC coefficients (default: 13)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 50)

## Model Architectures

### Baseline CNN
- 3 Conv1D layers (32 → 64 → 128 filters)
- BatchNormalization after each convolution
- MaxPooling for dimensionality reduction
- Dropout for regularization
- Dense layers with ReLU activation

### CNN-LSTM Hybrid
- 2 Conv1D layers for spatial feature extraction
- 2 LSTM layers (128 → 64 units) for temporal modeling
- Dropout for regularization
- Dense layers with softmax for classification

## Tips

- Use `librosa.get_samplerate()` to check audio properties
- The RAVDESS dataset is consistent, so all files are approximately 3-4 seconds
- Consider downsampling to 22050 Hz for faster processing
- Use `numpy` for efficient feature extraction across all files
- For faster testing, comment out Actor_19-24 folders in `train_emotion_recognition.py`
- Monitor GPU usage with `nvidia-smi` on NVIDIA systems
- TensorFlow may require CUDA 11.8+ for GPU acceleration

## Project Timeline

- **Nov 1**: Data preprocessing and exploration (Current Phase)
- **Nov 15**: Baseline CNN model development
- **Nov 21-25**: Feature comparison and optimization
- **Dec 8**: Final submission

## Team

- Shaik
- Kevin Wrenn

## References

RAVDESS Dataset: https://zenodo.org/record/1188976
