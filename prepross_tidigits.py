import os
import pathlib
import tensorflow as tf
import argparse
import numpy as np
import pickle
import logging
import requests
import tarfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def download_and_extract_dataset(url, extract_path):
    local_tar_path = extract_path / "tidigits.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_tar_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with tarfile.open(local_tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_path)
    os.remove(local_tar_path)

parser = argparse.ArgumentParser()
parser.add_argument('--drop_freq', help='dim frequency', required=False)
parser.add_argument('--drop_int', help='dim amplitude', required=False)
parser.add_argument('--feature', help='feature type (spec or mfcc)', required=False)

args = parser.parse_args()
drop_freq = float(args.drop_freq) if args.drop_freq else 0.0
drop_int = float(args.drop_int) if args.drop_int else 0.0
feature = args.feature if args.feature else 'spec'
DATASET_PATH = 'data/tidigits'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_dataset("https://catalog.ldc.upenn.edu/desc/addenda/LDC93S10.tgz", data_dir)

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

train_ds = train_ds.map(lambda audio, labels: (tf.squeeze(audio, axis=-1), labels), tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda audio, labels: (tf.squeeze(audio, axis=-1), labels), tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

def get_spectrogram(waveform, drop_int=None, drop_freq=None):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    if drop_int:
        spectrogram = tf.nn.dropout(spectrogram, rate=drop_int)
    if drop_freq:
        spectrogram = tf.nn.dropout(tf.transpose(spectrogram), rate=drop_freq)
        spectrogram = tf.transpose(spectrogram)
    return spectrogram[..., tf.newaxis]

def get_mfcc(waveform, sample_rate, drop_int=None, drop_freq=None, num_mel_bins=40, num_mfccs=13):
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)
    if drop_int:
        spectrogram = tf.nn.dropout(spectrogram, rate=drop_int)
    if drop_freq:
        spectrogram = tf.nn.dropout(tf.transpose(spectrogram), rate=drop_freq)
        spectrogram = tf.transpose(spectrogram)
    
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], sample_rate, 0.0, sample_rate / 2)
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]
    return mfccs[..., tf.newaxis]

def make_spec_ds(ds, feature, drop_int=None, drop_freq=None):
    if feature == 'spec':
        return ds.map(lambda audio, label: (get_spectrogram(audio, drop_int, drop_freq), label), num_parallel_calls=tf.data.AUTOTUNE)
    elif feature == 'mfcc':
        return ds.map(lambda audio, label: (get_mfcc(audio, 16000, drop_int, drop_freq), label), num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds, feature, drop_int, drop_freq)
val_spectrogram_ds = make_spec_ds(val_ds, feature, drop_int, drop_freq)
test_spectrogram_ds = make_spec_ds(test_ds, feature, drop_int, drop_freq)

save_dir = 'saved_datasets'
os.makedirs(save_dir, exist_ok=True)

tf.data.experimental.save(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
tf.data.experimental.save(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
tf.data.experimental.save(test_spectrogram_ds, os.path.join(save_dir, 'test_spectrogram_ds'))

print("Datasets saved successfully.")
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break
input_shape = example_spectrograms.shape[1:]

logging.info(f'Dimension of spectrogram {input_shape}')

