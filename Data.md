# Data Overview and Usage Guide

This README explains the structure and purpose of each dataset in the `data/` folder, and provides guidance for using the data in the next steps of the proposal.

---


## Directory Structure (SCC)

```
data/
├── Audio_Song_Actors_01-24 (Actors 1 to 17)/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── ...
│   └── Actor_17/
├── Audio_Song_Actors_01-24 (Actors 19 to 24)/
│   ├── Actor_19/
│   ├── ...
│   └── Actor_24/
├── Audio_Speech_Actors_01-24 (Actors 1 to 17)/
│   ├── Actor_01/
│   ├── ...
│   └── Actor_17/
├── Audio_Speech_Actors_01-24 (Actors 19 to 24)/
│   ├── Actor_19/
│   ├── ...
│   └── Actor_24/
├── Video_Song_ (Actors 1 to 17)/
│   ├── Actor_01/
│   ├── ...
│   └── Actor_17/
├── Video_Song_ (Actor 19 to 24)/
│   ├── Actor_19/
│   ├── ...
│   └── Actor_24/
├── Video_Speech_ (Actors 1 to 17)/
│   ├── Actor_01/
│   ├── ...
│   └── Actor_17/
├── Video_Speech_ (Actor 19 to 24)/
│   ├── Actor_19/
│   ├── ...
│   └── Actor_24/
```

## Key Points
- Each subfolder contains files for a specific actor.
- Audio and video data are split by actor groups (1–17 and 19–24).
- File names encode emotion and other metadata (see RAVDESS documentation).

## How to Use in Code
- Update your code to use the correct folder names, e.g.:
  - `data/Audio_Song_Actors_01-24 (Actors 1 to 17)/Actor_01/`
  - `data/Audio_Speech_Actors_01-24 (Actors 1 to 17)/Actor_01/`
- When listing actors, use:
  ```python
  song_dir = 'data/Audio_Song_Actors_01-24 (Actors 1 to 17)'
  speech_dir = 'data/Audio_Speech_Actors_01-24 (Actors 1 to 17)'
  actors = sorted([d for d in os.listdir(song_dir) if d.startswith('Actor_')])
  ```
- For video, use the corresponding `Video_Song_ (Actors 1 to 17)` and `Video_Speech_ (Actors 1 to 17)` folders.

## Example Code Update
```python
song_dir = 'data/Audio_Song_Actors_01-24 (Actors 1 to 17)'
speech_dir = 'data/Audio_Speech_Actors_01-24 (Actors 1 to 17)'
actors = sorted([d for d in os.listdir(song_dir) if d.startswith('Actor_')])
example_file = os.path.join(song_dir, 'Actor_01', os.listdir(os.path.join(song_dir, 'Actor_01'))[0])
y, sr = librosa.load(example_file, sr=None)
print(f'Loaded {example_file}')
print(f'Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds')
```

## Tips
- Always check the folder and file names on SCC before running your code.
- Use absolute or relative paths as needed for your environment.
- Document any changes to folder names or structure in your code and README.

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
