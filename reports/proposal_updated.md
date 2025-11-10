# "Hear Me Out" – Emotion Recognition from Voice

## Team Members

- Parin Shaik
- Kevin Wrenn

## Elevator Pitch

We will train a deep learning model to detect human emotions (happy, sad, angry, neutral) from short voice clips by comparing how different audio representations — MFCCs vs Mel-spectrograms — influence model performance and robustness.

## Context

Emotion recognition from speech is an emerging area in affective computing with applications in human–computer interaction, accessibility, and virtual assistants. Our project builds upon emotional speech datasets such as RAVDESS, which provide recordings of actors vocalizing sentences in distinct emotional tones. The datasets contain balanced samples across multiple speakers, genders, and emotional states, allowing for controlled training and evaluation.

We will derive both Mel-spectrograms and MFCCs from the same dataset and audio samples, ensuring consistency in data distribution and normalization.

### References

- RAVDESS: [Ryerson Audio-Visual Database](https://zenodo.org/record/1188976)
- CREMA-D (backup): [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- Emotional Speech Data: [HLT Singapore Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data)

## Methods

### Preprocessing

- Convert .wav files into Mel-spectrograms and MFCCs (both derived from the same samples)
- Normalize feature scales and apply augmentation techniques:
  - Pitch shift
  - Time stretch
  - Background noise

### Model Architectures

#### Baseline CNN

- 3–4 convolutional layers with ReLU activation and max pooling
- Trained on Mel-spectrograms for emotion classification

#### CNN-LSTM Hybrid (Custom)

1. CNN component:
   - Extracts 2D spatial features from the spectrogram

2. LSTM component:
   - Processes reshaped feature maps as time sequences
   - Captures temporal dynamics (changes in pitch, loudness, tone evolution)

3. Output layer:
   - Dense layers with softmax activation
   - Produces emotion probabilities

The hybrid architecture is designed to understand evolving emotional cues (e.g., rising pitch or intensity that signals anger).

#### Training Optimizations

- Early stopping
- Dropout layers
- Learning-rate scheduling

### Hyperparameter Optimization

- Tune the following using validation F1-score:
  - Batch size
  - Learning rate
  - Number of LSTM hidden units

### Data Sources

- **Primary**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Backup**: CREMA-D (used only if RAVDESS proves insufficient in sample diversity)
- All preprocessing will be done on a single dataset to ensure normalization consistency

### Technical Stack

#### Libraries

- PyTorch
- Keras
- torchaudio
- librosa
- scikit-learn
- matplotlib

#### References/Tutorials

- PyTorch Audio Classification Tutorial
- [Keras Audio Classification with Spectrograms](https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy)
- Librosa documentation for MFCC and Mel feature extraction

#### Development Tools

- ChatGPT and GitHub Copilot for syntax and debugging

## Innovation Highlights

Our project goes beyond standard tutorials by:

- Comparing feature extraction methods — MFCCs vs Mel-spectrograms — on identical data points
- Designing a custom CNN-LSTM hybrid architecture from scratch to capture temporal emotional nuances
- Testing real-world robustness by systematically introducing background noise and measuring degradation
- Visualizing interpretability via:
  - Confusion matrices
  - t-SNE embeddings (for dimensionality reduction)
  - Activation maps

## Project Timeline

| Date | Milestone | Deliverable |
|------|-----------|-------------|
| Nov 1 | Data loading, preprocessing, and visualization | Code notebook showing Mel-spectrograms and MFCCs for several samples |
| Nov 15 | TA Checkpoint Presentation — baseline CNN model | Trained CNN with validation metrics and example predictions |
| Nov 21–25 | Feature comparison + noise robustness experiments | CNN-LSTM hybrid results and comparative tables |
| Dec 8 | Final submission | Paper, slides, and demo video (live audio classification) |

## Evaluation Plan

### Metrics

- Accuracy, precision, recall, F1-score
- Confusion matrix across four emotion classes
- Noise robustness:
  - Measure accuracy under increasing noise levels (10%, 30%, 50%)
- Live demo capability:
  - Real-time emotion prediction from uploaded/recorded audio clips

### Experimental Design

#### Feature Representation Study

- MFCC vs Mel-spectrogram (from the same dataset)
- Compare training curves and generalization

#### Architecture Comparison

- CNN vs CNN-LSTM
- Vary convolutional kernel sizes and LSTM hidden dimensions

#### Robustness Analysis

- Evaluate model performance at different noise levels:
  - 0% (baseline)
  - 10% background noise
  - 30% background noise
  - 50% background noise
- Test each variation for:
  - Performance stability
  - Generalization capability

## Summary

"Hear Me Out" explores how deep neural networks perceive emotional tone in human speech by combining audio feature engineering, hybrid model design, and robustness analysis. Our results aim to demonstrate how temporal modeling and feature choice impact emotion recognition accuracy.