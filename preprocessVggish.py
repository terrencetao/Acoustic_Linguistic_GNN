import os
import argparse
import pathlib
import pickle
import numpy as np
import tensorflow as tf
import librosa
import torch
import torchaudio
from torchvggish import vggish, vggish_input
import logging
import sys
from pathlib import Path 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MIN_VGGISH_INPUT_LENGTH_SECONDS = 0.96


def squeeze(audio, label):
    audio = tf.squeeze(audio, axis=-1)
    return audio, label

def extract_mfcc(audio_np, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)

def extract_spectrogram(audio_np, sr=16000):
    spectrogram = librosa.feature.melspectrogram(y=audio_np, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram)
    return log_spectrogram.T.astype(np.float32)

def extract_vgg(audio_np, sr=22400):
    audio_np = librosa.util.fix_length(audio_np, size=sr)
    spectrogram = librosa.feature.melspectrogram(y=audio_np, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
    return log_spectrogram.astype(np.float32)

def extract_vggish(audio_np):
    try:
        # Ensure audio is at least 0.96s and not empty
        if len(audio_np) == 0:
            raise ValueError("Empty audio input")
            
        min_len = int(16000 * MIN_VGGISH_INPUT_LENGTH_SECONDS)
        if len(audio_np) < min_len:
            pad_width = min_len - len(audio_np)
            audio_np = np.pad(audio_np, (0, pad_width), mode='constant')
        else:
            audio_np = audio_np[:min_len]

        # Convert to float32 and check for NaN/inf
        audio_np = audio_np.astype(np.float32)
        if not np.isfinite(audio_np).all():
            raise ValueError("Audio contains NaN or inf values")

        # Generate examples and validate
        example = vggish_input.waveform_to_examples(audio_np, 16000)
        if example.numel() == 0:
            raise ValueError("VGGish example generation produced empty tensor")

        # Process through model
        model = vggish()
        model.eval()
        with torch.no_grad():
            embedding = model.forward(example)
            
        if embedding.numel() == 0:
            raise ValueError("VGGish model produced empty embedding")
            
        return embedding.numpy().astype(np.float32)
        
    except Exception as e:
        print(f"Error processing audio clip: {str(e)}")
        # Return zero array of expected shape as fallback
        return np.zeros((128,), dtype=np.float32)  # VGGish produces 128-dim embeddings

def extract_feature(audio, feature_type):
    try:
        audio_np = audio.numpy().squeeze()
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.numpy()
            
        # Check for silent/empty audio
        if len(audio_np) == 0 or np.max(np.abs(audio_np)) < 1e-6:
            raise ValueError("Silent or empty audio input")
            
        if feature_type == 'mfcc':
            return extract_mfcc(audio_np)
        elif feature_type == 'spectrogram':
            return extract_spectrogram(audio_np)
        elif feature_type == 'vgg':
            return extract_vgg(audio_np)
        elif feature_type == 'vggish':
            return extract_vggish(audio_np)
        else:
            raise ValueError("Invalid feature type")
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        # Return appropriate zero array based on feature type
        if feature_type == 'mfcc':
            return np.zeros((32, 13), dtype=np.float32)  # Typical MFCC shape
        elif feature_type == 'spectrogram':
            return np.zeros((128, 128), dtype=np.float32)  # Typical spectrogram shape
        elif feature_type == 'vgg':
            return np.zeros((128, 128, 1), dtype=np.float32)
        elif feature_type == 'vggish':
            return np.zeros((128,), dtype=np.float32)

def process_dataset(ds, feature_type):
    def extract_fn(audio, label):
        feature = tf.py_function(
            func=lambda a: extract_feature(a, feature_type),
            inp=[audio],
            Tout=tf.float32
        )
        # Ensure consistent output shapes
        if feature_type == 'vggish':
            feature.set_shape([128])  # VGGish produces 128-dim embeddings
        return feature, label

    return ds.map(extract_fn, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == '__main__':
    # Initialize argument parser with better help messages
    parser = argparse.ArgumentParser(description='Process audio features from mini speech commands dataset')
    parser.add_argument('--drop_freq', help='Frequency dimension to drop (optional)', required=False)
    parser.add_argument('--drop_int', help='Amplitude dimension to drop (optional)', required=False)
    parser.add_argument('--feature', 
                       help='Type of features to extract', 
                       required=True, 
                       choices=['mfcc', 'spectrogram', 'vgg', 'vggish'])
    parser.add_argument('--output_train', 
                       help='Directory to save processed training features',
                       required=True)
    parser.add_argument('--output_val', 
                       help='Directory to save processed validation features',
                       required=True)
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 1. Dataset preparation
        DATASET_PATH = 'data/mini_speech_commands'
        data_dir = pathlib.Path(DATASET_PATH)
        
        # Download dataset if needed with verification
        if not data_dir.exists():
            logger.info("Downloading dataset...")
            try:
                zip_path = tf.keras.utils.get_file(
                    'mini_speech_commands.zip',
                    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                    extract=True,
                    cache_dir='.', 
                    cache_subdir='data')
                
                # Verify extraction
                if not data_dir.exists() or len(list(data_dir.glob('*/*.wav'))) == 0:
                    raise RuntimeError("Dataset extraction failed - no WAV files found")
                logger.info("Dataset downloaded and extracted successfully")
            except Exception as e:
                logger.error(f"Failed to download dataset: {str(e)}")
                raise

        # 2. Load and validate commands
        commands = np.array([f.name for f in data_dir.iterdir() if f.is_dir()])
        if len(commands) == 0:
            raise ValueError("No command directories found in dataset")
        logger.info(f"Found commands: {commands}")

        # 3. Load dataset with error handling
        try:
            logger.info("Loading audio dataset...")
            train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
                directory=data_dir,
                batch_size=None,
                validation_split=0.2,
                seed=0,
                output_sequence_length=16000,
                subset='both')
            
            # Verify dataset loading
            if len(train_ds) == 0 or len(val_ds) == 0:
                raise ValueError("Empty training or validation dataset")
        except Exception as e:
            logger.error(f"Failed to load audio dataset: {str(e)}")
            raise

        # 4. Process labels
        label_names = np.array(train_ds.class_names)
        if not np.array_equal(np.sort(commands), np.sort(label_names)):
            raise ValueError("Commands and label names don't match")
        
        logger.info(f"Label names: {label_names}")
        
        # Save labels with error handling
        try:
            with open('label_names_google_command.pkl', 'wb') as f:
                pickle.dump(label_names, f)
        except Exception as e:
            logger.error(f"Failed to save label names: {str(e)}")
            raise

        # 5. Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
        val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

        # 6. Feature extraction with progress tracking
        logger.info(f"Extracting {args.feature} features...")
        
        # Create output directories if they don't exist
        Path(args.output_train).mkdir(parents=True, exist_ok=True)
        Path(args.output_val).mkdir(parents=True, exist_ok=True)
        
        try:
            train_features = process_dataset(train_ds, args.feature)
            val_features = process_dataset(val_ds, args.feature)
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

        # 7. Save features with error handling
        logger.info(f"Saving features to {args.output_train} and {args.output_val}")
        try:
            tf.data.experimental.save(train_features, args.output_train)
            tf.data.experimental.save(val_features, args.output_val)
            logger.info("Feature saving completed successfully")
        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

    logger.info("Processing completed successfully")