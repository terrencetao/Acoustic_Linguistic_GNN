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

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def drof_freq(spectrogram, drop_fraction):
  drop_fraction = 0.5  # Fraction of frequency bins to drop

# Calculate the number of frequencies to keep
  num_frequencies = spectrogram.shape[2]
  num_to_keep = int(num_frequencies * (1 - drop_fraction))
 
# Randomly select indices for the frequency dimension to keep
  frequency_indices = np.random.choice(num_frequencies, num_to_keep, replace=False)

# Sort the indices (optional, to maintain some order)
  frequency_indices = np.sort(frequency_indices)

# Downsample the spectrogram by selecting the random subset of frequencies
  spectrogram_downsampled = tf.gather(spectrogram, frequency_indices, axis=2)
  
  return spectrogram_downsampled

def get_spectrogram(waveform, drop_fraction):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  
  spectrogram = tf.abs(drof_freq(spectrogram, drop_fraction))
  
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def make_spec_ds(ds,dim):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio,dim), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

parser = argparse.ArgumentParser()
parser.add_argument('--drop_freq', help='dim frequency ', required=True)  

args = parser.parse_args()
drop_freq = float(args.drop_freq) 


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


train_spectrogram_ds = make_spec_ds(train_ds,drop_freq)
val_spectrogram_ds = make_spec_ds(val_ds,drop_freq)
test_spectrogram_ds = make_spec_ds(test_ds,drop_freq)



# Define the directory to save the datasets
save_dir = 'saved_datasets'
os.makedirs(save_dir, exist_ok=True)


# Save the datasets
tf.data.experimental.save(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
tf.data.experimental.save(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
tf.data.experimental.save(test_spectrogram_ds, os.path.join(save_dir, 'test_spectrogram_ds'))

print("Datasets saved successfully.")

