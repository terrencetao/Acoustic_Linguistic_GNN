import os
import librosa
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import pickle
from pathlib import Path
from tqdm import tqdm

# --- PARAMÈTRES ---
DATASET_PATH = 'data/TIMIT'  # <-- à adapter
SAVE_DIR = 'saved_datasets/timit'
TARGET_SAMPLING_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 128
FIXED_MFCC_SHAPE = (13, 32)
TRAIN_SPLIT = 0.8
SEED = 42

mfccs = []
labels = []

# --- EXTRAIRE LES SEGMENTS AUDIO ---
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
                    mfcc = librosa.feature.mfcc(
                        y=segment, sr=TARGET_SAMPLING_RATE,
                        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
                    )
                    mfcc_resized = tf.image.resize_with_pad(
                        mfcc[np.newaxis, ..., np.newaxis],
                        FIXED_MFCC_SHAPE[0],
                        FIXED_MFCC_SHAPE[1]
                    )[0].numpy()
                    mfccs.append(mfcc_resized)
                    labels.append(label)
                except Exception as e:
                    print(f"[!] Erreur MFCC: {e}")

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

    X_np = np.array(X)
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
    print("🚀 Démarrage du prétraitement TIMIT")
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

