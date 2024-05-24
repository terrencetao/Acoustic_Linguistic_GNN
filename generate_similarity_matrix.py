import tensorflow as tf
import os
import numpy as np
import pickle
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse 



import gensim.downloader as api
import numpy as np
import dgl
import torch
import pickle  # for saving label names
import eng_to_ipa as ipa
from sklearn.feature_extraction.text import CountVectorizer


# Function to extract spectrograms from the dataset and squeeze them
def extract_spectrograms(dataset):
    spectrograms = []
    labels = []
    for spectrogram_batch, label_batch in loaded_train_spectrogram_ds:
        for spectrogram in spectrogram_batch:
            spectrograms.append(tf.squeeze(spectrogram, axis=-1).numpy())
        for label in label_batch:
            labels.append(label.numpy())
    return spectrograms, labels



# Function to compute DTW distance between two spectrograms using dtaidistance
def compute_dtw_distance(spectrogram1, spectrogram2):
    distances = []
    for k in range(spectrogram1.shape[1]):  # iterate over frequency bins
        d = dtw.distance(spectrogram1[:, k], spectrogram2[:, k])
        distances.append(d)
    return np.mean(distances)

# Wrapper function for parallel processing
def compute_distance_for_pair(spectrograms, i, j):
    distance = compute_dtw_distance(spectrograms[i], spectrograms[j])
    return i, j, distance

# Function to compute DTW similarity matrix with parallel processing
def compute_dtw_similarity_matrix(spectrograms):
    num_spectrograms = len(spectrograms)
    similarity_matrix = np.zeros((num_spectrograms, num_spectrograms))
    
    # Create a list of pairs (i, j) to compute
    pairs = [(i, j) for i in range(num_spectrograms) for j in range(i, num_spectrograms)]
    
    # Use Parallel and delayed to parallelize the computation
    results = Parallel(n_jobs=-1)(delayed(compute_distance_for_pair)(spectrograms, i, j) for i, j in tqdm(pairs))
    
    # Fill in the similarity matrix with the computed distances
    for i, j, distance in results:
        similarity_matrix[i, j] = np.exp(-distance)  # Convert distance to similarity
        similarity_matrix[j, i] = similarity_matrix[i, j]  # Use symmetry
    
    return similarity_matrix
    
# Function to compute the median distance for each label group
def compute_median_distances(similarity_matrix, labels):
    unique_labels = np.unique(labels)
    median_distances = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        distances = []

        for i in indices:
            for j in indices:
                if i != j:
                    distances.append(similarity_matrix[i, j])

        median_distances[label] = np.median(distances)

    return median_distances
    
def compute_iqr_thresholds(similarity_matrix, labels):
    unique_labels = np.unique(labels)
    iqr_thresholds = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        distances = []

        for i in indices:
            for j in indices:
                if i != j:
                    distances.append(similarity_matrix[i, j])

        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        iqr_thresholds[label] = q1 - 1.5 * iqr  # Threshold for outliers

    return iqr_thresholds
    

import numpy as np

def filter_similarity_matrix(similarity_matrix, labels, threshold=0, alpha=2, k=None):
    # Make a copy of the similarity matrix to avoid modifying the original
    filtered_matrix = similarity_matrix.copy()
    
    # Get the size of the matrix
    n = similarity_matrix.shape[0]
    
    for i in range(n):
        # Get the indices of the values greater than the threshold
        valid_indices = np.where(similarity_matrix[i, :] > threshold)[0]
        
        if k is not None and len(valid_indices) > k:
            # Sort valid indices based on the similarity values in descending order
            sorted_indices = valid_indices[np.argsort(similarity_matrix[i, valid_indices])[-k:]]
        else:
            sorted_indices = valid_indices
        
        for j in range(n):
            if i == j or j not in sorted_indices:
                filtered_matrix[i, j] = 0
            elif labels[i] == labels[j]:
                filtered_matrix[i, j] = alpha * similarity_matrix[i, j]
    
    return filtered_matrix

    


parser = argparse.ArgumentParser()
parser.add_argument('--sub_unit', help='number for training', required=True)    
parser.add_argument('--num_n', help='number of neighbors for filering acoustic graph', required=True)
parser.add_argument('--ta', help='acoustic similarity threshold', required=True)
parser.add_argument('--tw', help='word similarity threshold', required=True)
args = parser.parse_args()
sub_units = int(args.sub_unit)    
 
# Define the directory where datasets are saved
save_dir = 'saved_datasets'

# Load the datasets
loaded_train_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'train_spectrogram_ds'))
loaded_val_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'val_spectrogram_ds'))
loaded_test_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'test_spectrogram_ds'))

print("Datasets loaded successfully.")


# Extract spectrograms
train_spectrograms, labels_train = extract_spectrograms(loaded_train_spectrogram_ds)
#val_spectrograms, labels_val = extract_spectrograms(loaded_val_spectrogram_ds)
#test_spectrograms, labels_test = extract_spectrograms(loaded_test_spectrogram_ds)



## Compute DTW similarity matrix for a subset (e.g., first 100 spectrograms)
subset_size = sub_units
subset_spectrograms = train_spectrograms[:subset_size]
subset_labels = labels_train[:subset_size]
similarity_matrix = compute_dtw_similarity_matrix(subset_spectrograms)

# Compute the median distances for each label group
median_distances = compute_median_distances(similarity_matrix, subset_labels)
#bornes_inferieures_iqr = compute_iqr_thresholds(similarity_matrix, subset_labels)
# Filter the similarity matrix based on the median thresholds and set diagonal to zero

medianes = np.array(list(median_distances.values()))
nan_mask = np.isnan(medianes)
filtered_similarity_matrix = filter_similarity_matrix(similarity_matrix, subset_labels, threshold=int(args.ta), k=int(args.num_n)))

print("Filtered similarity matrix computed successfully.")

# Convert subset_labels to a NumPy array
subset_labels = np.array(subset_labels)
# Append labels as an additional column
matrix_with_labels = np.hstack((subset_labels[:, np.newaxis], filtered_similarity_matrix))

# Save the matrix with labels
np.save('filtered_similarity_matrix_with_labels.npy', matrix_with_labels)
np.save('subset_spectrogram.npy', subset_spectrograms )


print("Filtered similarity matrix computed successfully.")












# Load label_names from the file to verify
with open('label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)
    


# Convert words to their phoneme representations
phoneme_words = [ipa.convert(word) for word in label_names]


# Initialize CountVectorizer
vectorizer = CountVectorizer(analyzer='char', token_pattern=r'[^ ]')

# Fit the vectorizer on the phoneme words and transform them to vectors
X = vectorizer.fit_transform(phoneme_words)


# Retrieve embeddings for each word in the list
word_embeddings = X.toarray()

# Compute the cosine similarity matrix
similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
norms = np.linalg.norm(word_embeddings, axis=1)
similarity_matrix = similarity_matrix / norms[:, np.newaxis]
similarity_matrix = similarity_matrix / norms[np.newaxis, :]

# Apply threshold
threshold = int(args.tw)
similarity_matrix[similarity_matrix < threshold] = 0

# Set diagonal to 0 to avoid self-loops
np.fill_diagonal(similarity_matrix, 0)

np.save('filtered_similarity_matrix_word.npy', similarity_matrix)
np.save('word_embedding.npy', word_embeddings )
print("Filtered similarity matrix for word label computed successfully.")
