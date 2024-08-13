import os
import pathlib
import zipfile
import requests
import shutil
import numpy as np
import tensorflow as tf
import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

digit_to_word = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine'
}

def sampler(spectrogram, dim, drop=None):
    drop_fraction = drop  
    num_dim = spectrogram.shape[dim]
    num_to_keep = int(num_dim * (1 - drop_fraction))
    indices = np.random.choice(num_dim, num_to_keep, replace=False)
    indices = np.sort(indices)
    spectrogram_downsampled = tf.gather(spectrogram, indices, axis=dim)
    return spectrogram_downsampled

def drop_out(spectrogram, drop_int=None, drop_freq=None):
    spectrogram_downsampled = tf.zeros_like(spectrogram)
    if drop_int:
        spectrogram_downsampled = sampler(spectrogram, dim=1, drop=drop_int)
    if drop_freq:
        spectrogram_downsampled = sampler(spectrogram_downsampled, dim=2, drop=drop_freq)
    return spectrogram_downsampled

def get_mfcc(waveform, sample_rate, drop_int=None, drop_freq=None, num_mel_bins=40, num_mfccs=13):
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)
    spectrogram = drop_out(spectrogram, drop_int, drop_freq)
    
    lower_edge_hertz = 0.0
    upper_edge_hertz = sample_rate / 2.0
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], sample_rate, lower_edge_hertz, upper_edge_hertz)
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]
    mfccs = mfccs[..., tf.newaxis]
    return mfccs

def get_spectrogram(waveform, drop_int=None, drop_freq=None):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = drop_out(spectrogram, drop_int, drop_freq)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def make_spec_ds(ds, feature, drop_int=None, drop_freq=None):
    if feature == 'spec':
        return ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio, drop_int, drop_freq), label),
            num_parallel_calls=tf.data.AUTOTUNE)
    elif feature == 'mfcc':
        return ds.map(
            map_func=lambda audio, label: (get_mfcc(audio, 16000, drop_int, drop_freq), label),
            num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def download_and_extract_dataset(url, extract_path):
    local_zip_path = extract_path / "spoken_digit.zip"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(local_zip_path)

def organize_files_into_directories(data_dir):
    for file in data_dir.glob('*.wav'):
        digit = file.stem.split('_')[0]
        label = digit_to_word[digit]
        label_dir = data_dir / label
        label_dir.mkdir(exist_ok=True)
        shutil.move(str(file), str(label_dir / file.name))

parser = argparse.ArgumentParser()
parser.add_argument('--drop_freq', help='dim frequency', required=False)  
parser.add_argument('--drop_int', help='dim amplitude', required=False) 
parser.add_argument('--feature', help='feature type (spec or mfcc)', required=False)

args = parser.parse_args()
drop_freq = float(args.drop_freq) 
drop_int = float(args.drop_int)
feature = args.feature
DATASET_PATH = 'data/spoken_digit'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_dataset("https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip", data_dir)
    extracted_dir = data_dir / "free-spoken-digit-dataset-master" / "recordings"
    for file in extracted_dir.glob('*.wav'):
        shutil.move(str(file), str(data_dir / file.name))
    shutil.rmtree(extracted_dir.parent)
    organize_files_into_directories(data_dir)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("Label names:", label_names)

with open('label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

train_spectrogram_ds = make_spec_ds(train_ds, feature, drop_int, drop_freq)
val_spectrogram_ds = make_spec_ds(val_ds, feature, drop_int, drop_freq)
test_spectrogram_ds = make_spec_ds(test_ds, feature, drop_int, drop_freq)

save_dir = 'saved_datasets/spoken_digit'
os.makedirs(save_dir, exist_ok=True)

tf.data.experimental.save(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
tf.data.experimental.save(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
tf.data.experimental.save(test_spectrogram_ds, os.path.join(save_dir, 'test_spectrogram_ds'))

print("Datasets saved successfully.")
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break
input_shape = example_spectrograms.shape[1:]

logging.info(f'Dimension of spectrogram {input_shape}')

