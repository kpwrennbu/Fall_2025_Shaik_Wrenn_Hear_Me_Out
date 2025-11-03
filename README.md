# Data Overview and Usage Guide

This README explains the structure and purpose of each dataset in the `data/` folder, and provides guidance for using the data in the next steps of the proposal.

---

## Directory Structure

- **Audio_Song_Actors_01-24/**
  - Contains audio song files for Actors 1–17 (subfolders: Actor_01 to Actor_17)
- **Audio_Song_Actors_01-24 (Actors 19 to 24)/**
  - Contains audio song files for Actors 19–24 (subfolders: Actor_19 to Actor_24)
- **Audio_Speech_Actors_01-24/**
  - Contains audio speech files for Actors 1–17 (subfolders: Actor_01 to Actor_17)
- **Audio_Speech_Actors_01-24_Cut/**
  - Contains cut/shortened audio speech files for Actors 19–24 (subfolders: Actor_19 to Actor_24)
- **Video_Song_/**
  - Contains video song files for Actors 1–17 (subfolders: Actor_01 to Actor_17)
- **Video_Song_Cut/**
  - Contains cut/shortened video song files for Actors 19–24 (subfolders: Actor_19 to Actor_24)
- **Video_Speech_/**
  - Contains video speech files for Actors 1–17 (subfolders: Actor_01 to Actor_17)
- **Video_Speech_Cut/**
  - Contains cut/shortened video speech files for Actors 19–24 (subfolders: Actor_19 to Actor_24)

---

## Data Types Explained

- **Song**: Actor performs a song with emotional expression.
- **Speech**: Actor speaks a sentence with emotional expression.
- **Cut**: Shortened or trimmed versions of the original files, useful for quick experiments or real-time demos.
- **Actors**: Each subfolder (e.g., Actor_01) contains files for a specific actor. Actor numbers correspond to unique individuals in the dataset.

---

## How to Use the Data

### 1. Data Loading
- Use libraries like `librosa` (for audio) and `opencv` (for video) to load files.
- Each actor's folder contains multiple files representing different emotions (happy, sad, angry, neutral).

### 2. Preprocessing
- For audio: Convert `.wav` files to Mel-spectrograms and MFCCs.
- For video: Extract frames or use video features if needed.
- Normalize features and apply augmentation (pitch shift, time stretch, background noise).

### 3. Labeling
- Labels are typically encoded in the filenames (check dataset documentation for exact format).
- Map each file to its corresponding emotion class.

### 4. Splitting
- Split data into training, validation, and test sets, ensuring balanced representation of actors and emotions.

### 5. Next Steps (Proposal)
- Use the processed features (Mel-spectrograms, MFCCs) as input to your deep learning models (CNN, CNN-LSTM).
- Compare model performance across different feature types and robustness to noise.
- Visualize results using confusion matrices, t-SNE, and activation maps.

---

## Tips
- Start with a small subset (e.g., a few actors) to validate your pipeline.
- Use the "Cut" datasets for rapid prototyping and real-time demo scenarios.
- Ensure consistent preprocessing for fair comparison between feature types.

---

## References
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- [Emotional Speech Data](https://github.com/HLTSingapore/Emotional-Speech-Data)

---

For questions or further details, refer to the main proposal or contact the project authors.
