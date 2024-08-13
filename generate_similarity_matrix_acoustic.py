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




import numpy as np
import dgl
import torch
import pickle 


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
    for k in range(spectrogram1.shape[0]):  # iterate over frequency bins
        d = dtw.distance(spectrogram1[k, :], spectrogram2[k, :])
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
    
def sim_matrix(method,subset_labels=None, subset_spectrograms=None):
   if method == 'dtw':
      similarity_matrix = compute_dtw_similarity_matrix(subset_spectrograms)
   elif method == 'fixed':
   # Convert labels list to a NumPy array
      labels_train_np = np.copy(subset_labels)

# Create a comparison matrix
      comparison_matrix = labels_train_np[:, None] == labels_train_np[None, :]

# Convert boolean matrix to integer matrix (0 and 1)
      similarity_matrix = comparison_matrix.astype(int)

   return similarity_matrix
    

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	  
	parser.add_argument('--sub_units', help='fraction of data', required=True)    
	parser.add_argument('--method', help='', required=True)
	parser.add_argument('--dataset', help='name of dataset', required=True)

	args = parser.parse_args()
	sub_units = int(args.sub_units)    
	 
	# Define the directory where datasets are saved
	save_dir = os.path.join('saved_datasets',args.dataset)

	# Load the datasets
	loaded_train_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'train_spectrogram_ds'))
	loaded_val_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'val_spectrogram_ds'))
	#loaded_test_spectrogram_ds = tf.data.experimental.load(os.path.join(save_dir, 'test_spectrogram_ds'))

	print("Datasets loaded successfully.")


	# Extract spectrograms
	train_spectrograms, labels_train = extract_spectrograms(loaded_train_spectrogram_ds)
	val_spectrograms, labels_val = extract_spectrograms(loaded_val_spectrogram_ds)
	#test_spectrograms, labels_test = extract_spectrograms(loaded_test_spectrogram_ds)

	  


	## Compute DTW similarity matrix for a subset (e.g., first 100 spectrograms)
	subset_size = sub_units
	subset_spectrograms = train_spectrograms[:subset_size]
	subset_val_spectrograms = val_spectrograms[:subset_size]
        
	subset_labels = labels_train[:subset_size]
	subset_val_labels = labels_val[:subset_size]
	# Convert subset_labels to a NumPy array
	subset_labels = np.array(subset_labels)
	subset_val_labels = np.array(subset_val_labels)


	# Compute the median distances for each label group
	#median_distances = compute_median_distances(similarity_matrix, subset_labels)
	#bornes_inferieures_iqr = compute_iqr_thresholds(similarity_matrix, subset_labels)
	# Filter the similarity matrix based on the median thresholds and set diagonal to zero

	#medianes = np.array(list(median_distances.values()))
	#nan_mask = np.isnan(medianes)
	#filtered_similarity_matrix = filter_similarity_matrix(similarity_matrix, subset_labels, threshold=int(args.ta), k=int(args.num_n))

	similarity_matrix = sim_matrix(method=args.method,  subset_labels=subset_labels, subset_spectrograms=subset_spectrograms)

	print(similarity_matrix)
	# Append labels as an additional column
	matrix_with_labels = np.hstack((subset_labels[:, np.newaxis], similarity_matrix))

	# Save the matrix with labels
	np.save('similarity_matrix_with_labels.npy', matrix_with_labels)
	np.save('subset_spectrogram.npy', subset_spectrograms )
	np.save('subset_label.npy', subset_labels )
	np.save('subset_val_spectrogram.npy', subset_val_spectrograms )
	np.save('subset_val_label.npy', subset_val_labels )



	print("Acoustic similarity matrix computed successfully.")

