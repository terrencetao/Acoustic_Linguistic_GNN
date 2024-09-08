import os
import zipfile
import pandas as pd
import shutil
import pathlib
import numpy as np
import tensorflow as tf
import argparse
import logging
import pickle
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Step 1: Unzip the dataset
def unzip_dataset(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Step 2: Load and parse the .xlsx file
def load_excel_mapping(xlsx_path):
    df = pd.read_excel(xlsx_path)
    return dict(zip(df['Id_word'], df['yemba_encoded']))

# Step 3: Create folders for each word
def create_word_folders(mapping, base_path):
    for word in mapping.values():
        word_folder = os.path.join(base_path, word)
        os.makedirs(word_folder, exist_ok=True)

# Step 4: Move audio files into corresponding word folders

def ensure_utf8(string):
    return string.encode('utf-8').decode('utf-8')

def move_and_convert_audio_to_mono(audio_folder, mapping, base_path):
    for file_name in os.listdir(audio_folder):
        if file_name.endswith('.wav'):
            id_word = file_name.split('_')[3]
            if int(id_word) in mapping:
                word = mapping[int(id_word)]
                word = ensure_utf8(word)
                destination_folder = os.path.join(base_path, word)
                audio_path = os.path.join(audio_folder, file_name)
                try:
                    audio = AudioSegment.from_wav(audio_path)
                    audio = audio.set_channels(1)  # Convert to mono
                    audio.export(os.path.join(destination_folder, file_name), format="wav")
                except CouldntDecodeError:
                    logging.error(f"Failed to decode {audio_path}. Skipping this file.")
     
# Spectrogram and MFCC processing functions
def sampler(spectrogram, dim, drop=None):
    if drop != 0.0:
        drop_fraction = drop  
        num_dim = spectrogram.shape[dim]
        num_to_keep = int(num_dim * (1 - drop_fraction))
        indices = np.random.choice(num_dim, num_to_keep, replace=False)
        indices = np.sort(indices)
        spectrogram_downsampled = tf.gather(spectrogram, indices, axis=dim)
    else:
        spectrogram_downsampled = spectrogram
    return spectrogram_downsampled

def drop_out(spectrogram, drop_int=None, drop_freq=None):
    spectrogram_downsampled = spectrogram
    if drop_int>0:
        spectrogram_downsampled = sampler(spectrogram, dim=1, drop=drop_int)
    if drop_freq>0:
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
            map_func=lambda audio, label: (get_mfcc(audio, 44100, drop_int, drop_freq), label),
            num_parallel_calls=tf.data.AUTOTUNE)

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels
    
def convert_to_mono(waveform):
    if waveform.shape[-1] > 1:
        waveform = tf.reduce_mean(waveform, axis=-1)
    print(waveform.shape)
    return waveform
    
def save_dataset(ds, path):
    path = ensure_utf8(path)
    try:
        tf.data.experimental.save(ds, path)
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")

def save_dataset_as_pickle(dataset, path):
    data_list = list(dataset)
    with open(path, 'wb') as f:
        pickle.dump(data_list, f)

def save_filenames_to_csv(filenames, save_dir, filename):
    csv_path = os.path.join(save_dir, filename)
    df = pd.DataFrame({
        'filenames': filenames,
    })
    df.to_csv(csv_path, index=False)
    logging.info(f"Filenames saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
    parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
    parser.add_argument('--feature', help='feature type: spec or mfcc', required=True)

    args = parser.parse_args()
    drop_freq = float(args.drop_freq) if args.drop_freq else 0.0
    drop_int = float(args.drop_int) if args.drop_int else 0.0
    feature = args.feature
    
    extract_path = 'data/yemba'
    xlsx_path = f'{extract_path}/corpus_words.xlsx'

    # Unzip the dataset
    unzip_dataset(f'{ extract_path}/audios.zip', f'{extract_path}/audios')

    # Load the Excel mapping
    word_mapping = load_excel_mapping(xlsx_path)
    
    # Create word folders
    create_word_folders(word_mapping, extract_path)

    # Move the audio files into the corresponding word folders
    move_and_convert_audio_to_mono(f'{extract_path}/audios/yemba_dataset', word_mapping, extract_path)

    logging.info("Dataset reformatted successfully.")
    
    # After moving and converting all files, delete the original audio folder
    shutil.rmtree(f'{extract_path}/audios/')

    # Assuming audio files are now in their respective word folders
    data_dir = pathlib.Path(extract_path)

    # Load the dataset
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)
    logging.info(f"Label names: {label_names}")
    
    # Save label_names to a file using pickle
    with open('label_names_yemba_command.pkl', 'wb') as f:
        pickle.dump(label_names, f)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    

    # Define the directory to save the datasets
    save_dir = 'saved_datasets/yemba_command'
    os.makedirs(save_dir, exist_ok=True)

    

    train_spectrogram_ds = make_spec_ds(train_ds, feature, drop_int, drop_freq)
    val_spectrogram_ds = make_spec_ds(val_ds, feature, drop_int, drop_freq)

    # Save the datasets as TFRecords
    save_dataset(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrogram_ds'))
    save_dataset(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrogram_ds'))
    
    # Save the datasets as Pickle files
    #save_dataset_as_pickle(train_spectrogram_ds, os.path.join(save_dir, 'train_spectrograms.pkl'))
    #save_dataset_as_pickle(val_spectrogram_ds, os.path.join(save_dir, 'val_spectrograms.pkl'))

    logging.info("Script executed successfully.")

