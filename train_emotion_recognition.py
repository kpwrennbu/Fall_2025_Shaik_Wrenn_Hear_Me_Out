"""
Emotion Recognition from Speech and Song - Training Pipeline
Project: "Hear Me Out"

This script implements the complete machine learning pipeline for emotion recognition,
including data loading, feature extraction, model training, and evaluation.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class EmotionRecognitionPipeline:
    """Complete pipeline for emotion recognition from audio."""
    
    def __init__(self, config):
        self.config = config
        self.emotion_list = sorted(list(set(config['target_emotions'])))
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotion_list)}
        self.idx_to_emotion = {i: e for e, i in self.emotion_to_idx.items()}
        
    def parse_ravdess_filename(self, filename):
        """Parse RAVDESS filename to extract emotion and metadata."""
        parts = filename.split('-')
        emotion_code = parts[2]
        emotion = self.config['emotion_map'].get(emotion_code, 'unknown')
        actor = int(parts[6].replace('.wav', ''))
        return emotion, actor
    
    def load_dataset(self):
        """Load audio files and labels from the dataset."""
        files = []
        labels = []
        data_dir = self.config['data_dir']
        target_emotions = self.config['target_emotions']
        
        actor_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('Actor_')])
        print(f'Found {len(actor_dirs)} actor directories')
        
        for actor_dir in actor_dirs:
            actor_path = os.path.join(data_dir, actor_dir)
            audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            
            for audio_file in audio_files:
                emotion, actor_id = self.parse_ravdess_filename(audio_file)
                if emotion in target_emotions:
                    files.append(os.path.join(actor_path, audio_file))
                    labels.append(emotion)
        
        print(f'Loaded {len(files)} audio files')
        print(f'Emotion distribution:')
        for emotion in target_emotions:
            count = labels.count(emotion)
            print(f'  {emotion}: {count}')
        
        return files, labels
    
    def extract_mfcc(self, audio_path):
        """Extract MFCC features from audio file."""
        try:
            y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.config['n_mfcc'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            return np.mean(mfcc, axis=1)
        except Exception as e:
            print(f'Error processing {audio_path}: {e}')
            return None
    
    def extract_melspectrogram(self, audio_path):
        """Extract Mel-spectrogram features from audio file."""
        try:
            y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.config['n_mels'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return np.mean(mel_spec_db, axis=1)
        except Exception as e:
            print(f'Error processing {audio_path}: {e}')
            return None
    
    def extract_all_features(self, audio_files):
        """Extract both MFCC and Mel-spectrogram features."""
        print('Extracting MFCC features...')
        mfcc_features = []
        for i, audio_file in enumerate(audio_files):
            if (i + 1) % 50 == 0:
                print(f'  Processed {i + 1}/{len(audio_files)} files')
            mfcc = self.extract_mfcc(audio_file)
            if mfcc is not None:
                mfcc_features.append(mfcc)
            else:
                mfcc_features.append(np.zeros(self.config['n_mfcc']))
        
        print('Extracting Mel-spectrogram features...')
        mel_features = []
        for i, audio_file in enumerate(audio_files):
            if (i + 1) % 50 == 0:
                print(f'  Processed {i + 1}/{len(audio_files)} files')
            mel = self.extract_melspectrogram(audio_file)
            if mel is not None:
                mel_features.append(mel)
            else:
                mel_features.append(np.zeros(self.config['n_mels']))
        
        return np.array(mfcc_features), np.array(mel_features)
    
    def prepare_data(self, X_mfcc, X_mel, y_labels):
        """Prepare and split data for training."""
        # Encode labels
        y_encoded = np.array([self.emotion_to_idx[e] for e in y_labels])
        y_categorical = to_categorical(y_encoded, num_classes=len(self.emotion_list))
        
        # Split data
        X_mfcc_train, X_mfcc_temp, y_train, y_temp = train_test_split(
            X_mfcc, y_categorical, test_size=0.4, random_state=42, stratify=y_encoded
        )
        X_mfcc_val, X_mfcc_test, y_val, y_test = train_test_split(
            X_mfcc_temp, y_temp, test_size=0.5, random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        X_mel_train, _, _, _ = train_test_split(
            X_mel, y_categorical, test_size=0.4, random_state=42, stratify=y_encoded
        )
        X_mel_val = X_mel[len(X_mfcc_train):len(X_mfcc_train) + len(X_mfcc_val)]
        X_mel_test = X_mel[len(X_mfcc_train) + len(X_mfcc_val):]
        
        # Normalize features
        scaler_mfcc = StandardScaler()
        X_mfcc_train = scaler_mfcc.fit_transform(X_mfcc_train)
        X_mfcc_val = scaler_mfcc.transform(X_mfcc_val)
        X_mfcc_test = scaler_mfcc.transform(X_mfcc_test)
        
        scaler_mel = StandardScaler()
        X_mel_train = scaler_mel.fit_transform(X_mel_train)
        X_mel_val = scaler_mel.transform(X_mel_val)
        X_mel_test = scaler_mel.transform(X_mel_test)
        
        print(f'Training set: {X_mfcc_train.shape}')
        print(f'Validation set: {X_mfcc_val.shape}')
        print(f'Test set: {X_mfcc_test.shape}')
        
        return (X_mfcc_train, X_mfcc_val, X_mfcc_test,
                X_mel_train, X_mel_val, X_mel_test,
                y_train, y_val, y_test)
    
    def build_baseline_cnn(self, input_shape):
        """Build baseline CNN model."""
        model = models.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.emotion_list), activation='softmax')
        ])
        return model
    
    def build_cnn_lstm_hybrid(self, input_shape):
        """Build CNN-LSTM hybrid model."""
        model = models.Sequential([
            layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            
            # CNN layers
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # LSTM layers
            layers.LSTM(128, return_sequences=True, activation='relu'),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.emotion_list), activation='softmax')
        ])
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        
        print(f'\n=== {model_name} ===')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test_labels, y_pred_labels, target_names=self.emotion_list))
        
        return accuracy, f1, y_pred_labels, y_test_labels


# Configuration
CONFIG = {
    'data_dir': 'data/Audio_Song_Actors_01-24_Actors_1_to_17',
    'n_mels': 128,
    'n_mfcc': 13,
    'sample_rate': 22050,
    'n_fft': 2048,
    'hop_length': 512,
    'emotion_map': {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    },
    'target_emotions': ['happy', 'sad', 'angry', 'neutral'],
    'batch_size': 32,
    'epochs': 50
}


if __name__ == '__main__':
    # Initialize pipeline
    pipeline = EmotionRecognitionPipeline(CONFIG)
    
    # Load data
    print('Loading dataset...')
    audio_files, emotion_labels = pipeline.load_dataset()
    
    # Extract features
    print('\nExtracting features...')
    X_mfcc, X_mel = pipeline.extract_all_features(audio_files)
    
    # Prepare data
    print('\nPreparing data...')
    X_mfcc_train, X_mfcc_val, X_mfcc_test, X_mel_train, X_mel_val, X_mel_test, y_train, y_val, y_test = \
        pipeline.prepare_data(X_mfcc, X_mel, emotion_labels)
    
    # Build models
    print('\nBuilding models...')
    os.makedirs('models', exist_ok=True)
    
    baseline_cnn = pipeline.build_baseline_cnn(X_mfcc_train.shape[1])
    baseline_cnn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    cnn_lstm = pipeline.build_cnn_lstm_hybrid(X_mel_train.shape[1])
    cnn_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train models
    print('\nTraining Baseline CNN...')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('models/baseline_cnn_best.h5', monitor='val_accuracy', save_best_only=True)
    ]
    baseline_cnn.fit(X_mfcc_train, y_train, validation_data=(X_mfcc_val, y_val),
                     epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                     callbacks=callbacks, verbose=1)
    
    print('\nTraining CNN-LSTM Hybrid...')
    cnn_lstm.fit(X_mel_train, y_train, validation_data=(X_mel_val, y_val),
                 epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                 callbacks=callbacks, verbose=1)
    
    # Evaluate models
    print('\n' + '='*50)
    print('MODEL EVALUATION')
    print('='*50)
    
    baseline_acc, baseline_f1, baseline_pred, baseline_test = \
        pipeline.evaluate_model(baseline_cnn, X_mfcc_test, y_test, 'Baseline CNN (MFCC)')
    
    lstm_acc, lstm_f1, lstm_pred, lstm_test = \
        pipeline.evaluate_model(cnn_lstm, X_mel_test, y_test, 'CNN-LSTM Hybrid (Mel-Spectrogram)')
    
    # Save models
    baseline_cnn.save('models/baseline_cnn_mfcc.h5')
    cnn_lstm.save('models/cnn_lstm_mel.h5')
    
    print('\n✓ Models saved successfully!')
