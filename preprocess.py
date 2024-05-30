import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def sampler(spectrogram, dim, drop=None ):
  drop_fraction = drop  
# Calculate the number of dim to keep
  num_dim = spectrogram.shape[dim]
  num_to_keep = int(num_dim * (1 - drop_fraction))
 
# Randomly select indices for the dim dimension to keep
  indices = np.random.choice(num_dim, num_to_keep, replace=False)

# Sort the indices (optional, to maintain some order)
  indices = np.sort(indices)

# Downsample the spectrogram by selecting the random subset of dim
  spectrogram_downsampled = tf.gather(spectrogram, indices, axis=dim)
  
  return spectrogram_downsampled

def drop_out(spectrogram, drop_int=None, drop_freq=None ):
  spectrogram_downsampled =  tf.zeros_like
  if drop_int:
     spectrogram_downsampled =  sampler(spectrogram, dim=1, drop=drop_int )
  if drop_freq:
     spectrogram_downsampled = sampler(spectrogram_downsampled, dim=2, drop=drop_freq )
    
  return spectrogram_downsampled
  
def get_spectrogram(waveform, drop_int=None, drop_freq=None):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  
  spectrogram = tf.abs(drop_out(spectrogram, drop_int, drop_freq))
  
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def make_spec_ds(ds,drop_int=None, drop_freq=None):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio, drop_int, drop_freq), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

parser = argparse.ArgumentParser()
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 

args = parser.parse_args()
drop_freq = float(args.drop_freq) 
drop_int = float(args.drop_int)

DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')
      
      
      
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
print("label names:", label_names)

# Save label_names to a file using pickle
with open('label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)






train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)


test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)


train_spectrogram_ds = make_spec_ds(train_ds, drop_int, drop_freq)
val_spectrogram_ds = make_spec_ds(val_ds, drop_int, drop_freq)
test_spectrogram_ds = make_spec_ds(test_ds, drop_int, drop_freq)



# Define the directory to save the datasets
save_dir = 'saved_datasets'
os.makedirs(save_dir, exist_ok=True)


# Save the datasets
tf.data.experimental.save(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
tf.data.experimental.save(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
tf.data.experimental.save(test_spectrogram_ds, os.path.join(save_dir, 'test_spectrogram_ds'))

print("Datasets saved successfully.")
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break
input_shape = example_spectrograms.shape[1:]
logging.info(f'dimension of spectrogram {input_shape}')

