import tensorflow as tf
import numpy as np
import librosa
import os
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
from transformers import WavLMModel, Wav2Vec2Processor
import tensorflow_hub as hub
import numpy as np
import logging
from typing import Tuple, Optional
from transformers import HubertModel, Wav2Vec2Processor
import torch

from torchvggish import vggish, vggish_input

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Chargement des données audio
def load_data(data_path, batch_size=1):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_path,
        batch_size=batch_size,
        output_sequence_length=16000,
        validation_split=0.2,
        seed=0,
        subset='both'
    )
    label_names = train_ds.class_names
    return train_ds, val_ds, label_names

# MFCC extraction
def make_mfcc_ds(ds, sr=16000, n_mfcc=13, n_fft=256, hop_length=160, debug=False):
    """
    Create MFCC features dataset from audio dataset using librosa.
    Returns mean along time axis and handles errors gracefully.
    
    Args:
        ds: TensorFlow dataset of (audio, label) pairs
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        debug: Enable debug logging
        
    Returns:
        Dataset of (mfcc_features, label) pairs
    """
    def extract_fn(audio, label):
        try:
            # Convert tensor to numpy array
            audio_np = audio.numpy()
            
            # Extract MFCCs using librosa
            mfccs = librosa.feature.mfcc(
                y=audio_np, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Take mean along time axis
            mfccs_mean = np.mean(mfccs, axis=1)
            
            # Check for NaN/inf and zero them
            if np.any(np.isnan(mfccs_mean)) or np.any(np.isinf(mfccs_mean)):
                if debug:
                    logger.warning(f"NaN/Inf detected in MFCC features, zeroing. Shape: {mfccs_mean.shape}")
                mfccs_mean = np.nan_to_num(mfccs_mean, nan=0.0, posinf=0.0, neginf=0.0)
                
            if debug:
                logger.info(f"Generated MFCC features of shape: {mfccs_mean.shape}")
                
            return mfccs_mean, label
            
        except Exception as e:
            if debug:
                logger.error(f"Error processing audio for MFCC: {str(e)}")
            # Return zero vector on error
            return np.zeros(n_mfcc), label
    
    return ds.map(
        lambda audio, label: tf.py_function(
            extract_fn, 
            [audio, label], 
            [tf.float32, label.dtype]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def make_spec_ds(ds, sr=16000, n_fft=256, hop_length=160, n_mels=64, debug=False):
    """
    Create Mel-spectrogram features dataset from audio dataset using librosa.
    Returns mean along time axis and handles errors gracefully.
    
    Args:
        ds: TensorFlow dataset of (audio, label) pairs
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of Mel bands
        debug: Enable debug logging
        
    Returns:
        Dataset of (mel_features, label) pairs
    """
    def extract_fn(audio, label):
        try:
            # Convert tensor to numpy array
            audio_np = audio.numpy()
            
            # Extract Mel spectrogram using librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # Convert to dB and take mean along time axis
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_mean = np.mean(mel_spec_db, axis=1)
            
            # Check for NaN/inf and zero them
            if np.any(np.isnan(mel_spec_mean)) or np.any(np.isinf(mel_spec_mean)):
                if debug:
                    logger.warning(f"NaN/Inf detected in Mel features, zeroing. Shape: {mel_spec_mean.shape}")
                mel_spec_mean = np.nan_to_num(mel_spec_mean, nan=0.0, posinf=0.0, neginf=0.0)
                
            if debug:
                logger.info(f"Generated Mel features of shape: {mel_spec_mean.shape}")
                
            return mel_spec_mean, label
            
        except Exception as e:
            if debug:
                logger.error(f"Error processing audio for Mel spectrogram: {str(e)}")
            # Return zero vector on error
            return np.zeros(n_mels), label
    
    return ds.map(
        lambda audio, label: tf.py_function(
            extract_fn, 
            [audio, label], 
            [tf.float32, label.dtype]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
# TRILL embedding
def make_trill_ds(dataset):
    trill_model_handle = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
    trill_model = hub.load(trill_model_handle)

    def extract_fn(waveform, label):
        try:
            print(f"\nInput shape: {waveform.shape}, label: {label}")
            
            waveform = tf.squeeze(waveform)
            waveform = tf.cast(waveform, tf.float32)
            max_val = tf.reduce_max(tf.abs(waveform))
            max_val = tf.maximum(max_val, 1e-6)  # Évite la division par 0
            waveform = waveform / max_val
            waveform = tf.reshape(waveform, [1, -1])
            
            embedding = trill_model(waveform, sample_rate=16000)['embedding']
            embedding = tf.reduce_mean(embedding, axis=1)
            
            if tf.reduce_any(tf.math.is_nan(embedding)):
                print("⚠️ NaN in embedding! Replacing with zeros.")
                embedding = tf.zeros_like(embedding)
                
            print(f"Output shape: {embedding.shape}")
            return embedding, label
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return tf.zeros([1, 512]), label  # Fallback output

    return dataset.map(extract_fn, num_parallel_calls=tf.data.AUTOTUNE)

def make_wav2vec_ds(dataset):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    import numpy as np

    # Initialize processor and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    def extract_fn(audio, label):
        try:
            # Convert to numpy and process
            audio_np = audio.numpy().squeeze()
            
            # Process through Wav2Vec2
            inputs = processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert to numpy and return
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.astype(np.float32), label
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return np.zeros((1, 768), dtype=np.float32), label

    return dataset.map(
        lambda x, y: tf.py_function(
            extract_fn,
            [x, y],
            [tf.TensorSpec(shape=(None, 768), dtype=tf.float32), y.dtype]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )               
                      
                      


def make_wavlm_ds(dataset):
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    import torch
    import numpy as np

    # Initialize feature extractor and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
    model.eval()

    def extract_fn(audio, label):
        try:
            # Convert to numpy and process
            audio_np = audio.numpy().squeeze()
            
            # Process through feature extractor
            inputs = feature_extractor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert to numpy and return mean embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.astype(np.float32), label
            
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            return np.zeros((1, 768), dtype=np.float32), label

    return dataset.map(
        lambda x, y: tf.py_function(
            extract_fn,
            [x, y],
            [tf.TensorSpec(shape=(None, 768), dtype=tf.float32), y.dtype]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )                       
                       
                       
                       


def make_hubert_ds(dataset):
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    import torch
    import numpy as np

    # Initialize feature extractor and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()

    def extract_fn(audio, label):
        try:
            # Convert to numpy and process
            audio_np = audio.numpy().squeeze()
            
            # Process through feature extractor
            inputs = feature_extractor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert to numpy and return mean embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.astype(np.float32), label
            
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            return np.zeros((1, 768), dtype=np.float32), label

    return dataset.map(
        lambda x, y: tf.py_function(
            extract_fn,
            [x, y],
            [tf.TensorSpec(shape=(None, 768), dtype=tf.float32), y.dtype]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )           
                       






def make_vggish_ds(dataset):
    vggish_model_handle = "https://tfhub.dev/google/vggish/1"
    vggish_model = hub.load(vggish_model_handle)

    def extract_fn(batch_inputs, batch_labels):
        embeddings_list = []
        labels_list = []

        for i in range(batch_inputs.shape[0]):
            try:
                input_data = tf.squeeze(batch_inputs[i])
                input_data = tf.cast(input_data, tf.float32)

                # Exemple de normalisation si nécessaire, à adapter selon l'entrée VGGish attendue
                max_val = tf.reduce_max(tf.abs(input_data))
                input_data = input_data / tf.maximum(max_val, 1e-6)

                # Appliquer VGGish
                embeddings = vggish_model(input_data)
                # embeddings shape peut dépendre du modèle, ajuster ici pour obtenir un vecteur fixe
                # Par exemple, faire une moyenne si embeddings est temporel
                if len(embeddings.shape) > 1:
                    mean_embedding = tf.reduce_mean(embeddings, axis=0)
                else:
                    mean_embedding = embeddings

                # Vérification NaN
                if tf.reduce_any(tf.math.is_nan(mean_embedding)):
                    print("⚠️ NaN dans VGGish embedding. Remplacement par des zéros.")
                    mean_embedding = tf.zeros_like(mean_embedding)

                embeddings_list.append(mean_embedding)
                labels_list.append(batch_labels[i])

            except Exception as e:
                print(f"❌ VGGish erreur sur l'exemple {i}: {e}")
                embeddings_list.append(tf.zeros([128], dtype=tf.float32))  # Exemple de taille embedding VGGish
                labels_list.append(batch_labels[i])

        return tf.stack(embeddings_list), tf.stack(labels_list)

    return dataset.map(
        lambda x, y: tf.py_function(
            extract_fn, [x, y], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def make_yamnet_ds(dataset):
    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)

    def extract_fn(batch_waveforms, batch_labels):
        embeddings_list = []
        labels_list = []

        for i in range(batch_waveforms.shape[0]):
            try:
                waveform = tf.squeeze(batch_waveforms[i])
                waveform = tf.cast(waveform, tf.float32)

                # Normalisation entre -1 et 1
                max_val = tf.reduce_max(tf.abs(waveform))
                waveform = waveform / tf.maximum(max_val, 1e-6)

                # Appliquer YAMNet
                scores, embeddings, spectrogram = yamnet_model(waveform)
                mean_embedding = tf.reduce_mean(embeddings, axis=0)

                # Vérification NaN
                if tf.reduce_any(tf.math.is_nan(mean_embedding)):
                    print("⚠️ NaN dans YAMNet embedding. Remplacement par des zéros.")
                    mean_embedding = tf.zeros_like(mean_embedding)

                embeddings_list.append(mean_embedding)
                labels_list.append(batch_labels[i])

            except Exception as e:
                print(f"❌ YAMNet erreur sur l'exemple {i}: {e}")
                embeddings_list.append(tf.zeros([1024], dtype=tf.float32))
                labels_list.append(batch_labels[i])

        return tf.stack(embeddings_list), tf.stack(labels_list)

    return dataset.map(
        lambda x, y: tf.py_function(
            extract_fn, [x, y], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
