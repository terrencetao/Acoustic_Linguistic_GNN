import os
import librosa
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import pickle
from pathlib import Path
from tqdm import tqdm

# === NOUVEAU ===
import torch
import torchaudio
import tensorflow_hub as hub
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# --- PARAMÈTRES ---
DATASET_PATH = 'data/TIMIT'
SAVE_DIR = 'saved_datasets/timit'
TARGET_SAMPLING_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 128
FIXED_MFCC_SHAPE = (13, 32)
TRAIN_SPLIT = 0.8
SEED = 42
REPRESENTATION_TYPE = 'mfcc'  # mfcc | wav2vec | vgg | trill

mfccs = []
labels = []

# --- CHARGEMENT DES MODÈLES ---
if REPRESENTATION_TYPE == 'wav2vec':
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    wav2vec_model = bundle.get_model()
elif REPRESENTATION_TYPE == 'trill':
    trill_model = hub.load("https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3")
elif REPRESENTATION_TYPE == 'vgg':
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer("block5_pool").output)

# --- FONCTIONS DE REPRÉSENTATION ---
def extract_features(segment, sr):
    if REPRESENTATION_TYPE == 'mfcc':
        mfcc = librosa.feature.mfcc(
            y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        return tf.image.resize_with_pad(mfcc[np.newaxis, ..., np.newaxis], FIXED_MFCC_SHAPE[0], FIXED_MFCC_SHAPE[1])[0].numpy()

    elif REPRESENTATION_TYPE == 'wav2vec':
        if sr != bundle.sample_rate:
            segment = librosa.resample(segment, sr, bundle.sample_rate)
        waveform = torch.tensor(segment).float().unsqueeze(0)
        with torch.inference_mode():
            features = wav2vec_model(waveform)[0]  # (1, T, 768)
        return features.mean(dim=1).squeeze(0).numpy()  # (768,)

    elif REPRESENTATION_TYPE == 'trill':
        if sr != 16000:
            segment = librosa.resample(segment, sr, 16000)
        wav = tf.convert_to_tensor(segment, dtype=tf.float32)
        trill_out = trill_model({'audio': wav, 'sample_rate': 16000})
        return trill_out['embedding'].numpy().mean(axis=0)  # (2048,)

    elif REPRESENTATION_TYPE == 'vgg':
        S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_resized = tf.image.resize(S_dB[..., np.newaxis], (224, 224))
        S_rgb = tf.image.grayscale_to_rgb(S_resized)
        S_rgb = preprocess_input(S_rgb.numpy())
        features = vgg_model.predict(S_rgb[np.newaxis, ...], verbose=0)
        return features.squeeze()  # (7, 7, 512)

# --- EXTRACTION DES SEGMENTS ---
def extract_words_and_labels(dataset_path):
    phn_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.WRD'):
                phn_files.append(os.path.join(root, file))

    for phn_path in tqdm(phn_files, desc="🔍 Extraction des segments audio"):
        wav_path = phn_path.replace('.WRD', '.WAV')
        if not os.path.exists(wav_path):
            continue

        audio, sr = sf.read(wav_path)
        if sr != TARGET_SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLING_RATE)

        with open(phn_path, 'r') as f:
            for line in f:
                start, end, label = line.strip().split()
                start, end = int(start), int(end)
                segment = audio[start:end]
                if len(segment) < N_FFT:
                    continue
                try:
                    features = extract_features(segment, sr)
                    mfccs.append(features)
                    labels.append(label)
                except Exception as e:
                    print(f"[!] Erreur feature extraction: {e}")

# --- ENCODAGE DES LABELS ---
def encode_labels(labels):
    print("🔠 Encodage des labels...")
    label_names = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(label_names)}
    encoded = [label2idx[label] for label in tqdm(labels, desc="🔢 Encodage des labels")]
    return encoded, label_names

# --- DIVISION TRAIN/VAL ---
def split_dataset(X, y, train_ratio=TRAIN_SPLIT):
    print("🔀 Division des données train/val...")
    random.seed(SEED)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_idx = int(len(X) * train_ratio)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_np = np.array(X, dtype=object)
    y_np = np.array(y)

    train_data = tf.data.Dataset.from_tensor_slices((X_np[train_idx], y_np[train_idx]))
    val_data = tf.data.Dataset.from_tensor_slices((X_np[val_idx], y_np[val_idx]))
    return train_data, val_data

# --- SAUVEGARDE ---
def save_datasets(train_ds, val_ds, save_dir):
    print("💾 Sauvegarde des jeux de données...")
    tf.data.experimental.save(train_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
    tf.data.experimental.save(val_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
    print("✅ Jeux de données sauvegardés avec tf.data.experimental.save")

def save_label_names(label_names, save_dir):
    print("💾 Sauvegarde des noms de labels...")
    with open(os.path.join(save_dir, 'labels_names.pkl'), 'wb') as f:
        pickle.dump(label_names, f)
    print("✅ labels_names.pkl sauvegardé")

# --- PIPELINE PRINCIPAL ---
def preprocess_timit():
    print(f"🚀 Démarrage du prétraitement TIMIT ({REPRESENTATION_TYPE})")
    extract_words_and_labels(DATASET_PATH)
    print(f"🎧 Total des segments extraits : {len(mfccs)}")

    encoded_labels, label_names = encode_labels(labels)
    print(f"📚 Labels cibles : {label_names}")

    train_ds, val_ds = split_dataset(mfccs, encoded_labels)
    os.makedirs(SAVE_DIR, exist_ok=True)

    save_datasets(train_ds, val_ds, SAVE_DIR)
    save_label_names(label_names, SAVE_DIR)
    print("✅ Pipeline terminé avec succès.")

# --- EXÉCUTION ---
if __name__ == '__main__':
    preprocess_timit()

