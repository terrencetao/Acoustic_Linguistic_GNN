import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='number of epochs', required=True)
parser.add_argument('--dataset', help='name of dataset', required=True)
args = parser.parse_args()

# Load the datasets
save_dir = os.path.join('saved_datasets',args.dataset)

# Load label_names from the file to verify
with open('label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)
    
label_names = list(label_names)    

train_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'train_spectrogram_ds'))
val_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'val_spectrogram_ds'))
#test_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'test_spectrogram_ds'))


print("Datasets loaded successfully.")

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break


input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

EPOCHS = int(args.epochs)
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

model.save('models/model.keras')
