import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
from tensorflow.keras import layers, models
from utils import load_data, make_spec_ds, make_mfcc_ds, make_wav2vec_ds, make_trill_ds
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive le GPU

def build_simple_dense_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_feature_extractor(ffn_model):
    # Extrait la sortie de la couche dense_3
    return models.Model(inputs=ffn_model.input,
                        outputs=ffn_model.get_layer('dense_3').output)


def extract_feature(feature_type, dataset, ffn_model=None):
    if feature_type == 'mfcc':
        return make_mfcc_ds(dataset)
    elif feature_type == 'wav2vec':
        return make_wav2vec_ds(dataset)
    elif feature_type == 'trill':
        return make_trill_ds(dataset)
    elif feature_type == 'vgg':
        return make_spec_ds(dataset, 'vgg')
    elif feature_type == 'vgg_dense3' and ffn_model is not None:
        return make_vgg_ds(dataset, ffn_model)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='name of dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--feature', type=str, default='mfcc', choices=['mfcc', 'wav2vec', 'trill', 'vgg'], help='Feature type to extract')
    args = parser.parse_args()

    data_path = args.data_path
    feature_type = args.feature

    # Chargement du dataset
    train_ds, val_ds, label_names = load_data(data_path)
    num_classes = len(label_names)

    # Étape 1 : extraire les features
    print(f"🧪 Extracting features using: {feature_type}")
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    train_features_ds = extract_feature(feature_type, train_ds).cache().prefetch(tf.data.AUTOTUNE)
    val_features_ds = extract_feature(feature_type, val_ds).cache().prefetch(tf.data.AUTOTUNE)
    batch_size = 32
    print(f"Original train samples: {len(train_ds) * batch_size}")  # Should be ~8000
    print(f"Processed train samples: {train_features_ds.cardinality()}")
    
    # Étape 2 : Entraîner FFN si 'vgg'
    if feature_type == 'vgg':
        for spectrograms, labels in train_features_ds.take(1):
            input_shape = spectrograms.shape[1:]

        vgg_ffn_model = build_simple_dense_model(input_shape, num_classes)
        vgg_ffn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("🧠 Training FFN on VGG features...")
        vgg_ffn_model.fit(
            train_features_ds,
            validation_data=val_features_ds,
            epochs=10
        )

        # Étape 3 : extraire les features de la couche dense_3
        print("🔍 Extracting features from FFN hidden layer (dense_3)...")
        train_features_ds = extract_feature('vgg_dense3', train_features_ds, ffn_model=vgg_ffn_model)
        val_features_ds = extract_feature('vgg_dense3', val_features_ds, ffn_model=vgg_ffn_model)

    print("✅ Feature extraction complete.")
      
    # === SAUVEGARDE ===
    save_base_dir = f'saved_datasets/{args.dataset}'
    save_dir = os.path.join(save_base_dir, feature_type)
    os.makedirs(save_dir, exist_ok=True)
    
    
    train_features_ds = train_features_ds.cache()  # Cache after extraction
    val_features_ds = val_features_ds.cache()

    tf.data.Dataset.save(train_features_ds , os.path.join(save_dir, 'train_spectrogram_ds'))
    tf.data.Dataset.save(val_features_ds, os.path.join(save_dir, 'val_spectrogram_ds'))

    # Sauvegarde des labels
    # Save label_names to a file using pickle
    with open(f'label_names_{args.dataset}.pkl', 'wb') as f:
         pickle.dump(label_names, f)

# Log dimension
    for example_features, example_labels in train_features_ds.take(1):
        input_shape = example_features.shape[1:]
        break

    logging.info(f"Features ({feature_type}) saved to: {save_dir}")
    logging.info(f"Input feature shape: {input_shape}")



if __name__ == '__main__':
    main()

