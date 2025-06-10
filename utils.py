import tensorflow as tf
import numpy as np
import librosa
import os
from transformers import Wav2Vec2Processor, TFWav2Vec2Model
import tensorflow_hub as hub

# Chargement des données audio
def load_data(data_path, batch_size=32):
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
def make_mfcc_ds(dataset):
    def extract_fn(wave, label):
        wave = tf.squeeze(wave, axis=-1)
        spectrogram = tf.signal.stft(wave, frame_length=640, frame_step=320)
        magnitude = tf.abs(spectrogram)
        mel_spec = tf.tensordot(magnitude, tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=40,
            num_spectrogram_bins=magnitude.shape[-1],
            sample_rate=16000,
            lower_edge_hertz=20.0,
            upper_edge_hertz=4000.0
        ), 1)
        log_mel_spec = tf.math.log(mel_spec + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spec)[..., :13]
        return mfccs, label

    return dataset.map(extract_fn)

# Spectrogram extraction (log-magnitude)
def make_spec_ds(dataset):
    def extract_fn(wave, label):
        wave = tf.squeeze(wave, axis=-1)
        spectrogram = tf.signal.stft(wave, frame_length=640, frame_step=320)
        magnitude = tf.abs(spectrogram)
        log_spec = tf.math.log(magnitude + 1e-6)
        log_spec = tf.expand_dims(log_spec, -1)  # channel dim
        return log_spec, label
    return dataset.map(extract_fn)

# TRILL embedding
def make_trill_ds(dataset):
    trill_model_handle = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
    trill_model = hub.load(trill_model_handle)

    def extract_fn(waveform, label):
        waveform = tf.squeeze(waveform)
        waveform = tf.cast(waveform, tf.float32)
        waveform = waveform / tf.reduce_max(tf.abs(waveform))  # normalize
        waveform = tf.reshape(waveform, [1, -1])  # [1, time]
        embedding = trill_model(waveform, sample_rate=16000)['embedding']  # [1, T, D]
        embedding = tf.reduce_mean(embedding, axis=1)  # [1, D]
        
        return embedding, label

    return dataset.map(extract_fn)

# Wav2Vec2 with Hugging Face
def make_wav2vec_ds(dataset):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def extract_fn(audio, label):
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        audio_np = audio.numpy()

        inputs = processor(audio_np, sampling_rate=16000, return_tensors="tf")
        outputs = model(inputs.input_values).last_hidden_state  # [batch, time, dim]
        mean_embedding = tf.reduce_mean(outputs, axis=1)  # [batch, dim]
        return mean_embedding[0], label

    return dataset.map(lambda x, y: tf.py_function(extract_fn, [x, y], [tf.float32, tf.int64]))


