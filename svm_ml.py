from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
from sklearn import svm
import logging 
import numpy as np
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to train and evaluate SVM and return accuracy
def train_evaluate_svm( X_train, X_test, y_train, y_test):
    #X_train, X_test, y_train, y_test = train_test_split_data(embeddings, labels)
    
    # Initialize the SVM model
    clf = svm.SVC(kernel='linear')
    # Train the model
    clf.fit(X_train, y_train)
    # Predict on test data
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
    
# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms
      
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', help='number of epochs', required=True)
	parser.add_argument('--dataset', help='name of dataset', required=False)
	parser.add_argument('--method_sim', help='', required=True)
	parser.add_argument('--sub_units', help='fraction of data', required=True)  
	
	args = parser.parse_args()
	sub_units = args.sub_units
     
	logging.info(f'Train and evaluate SVM for spectrogram embeddings')

	matrix_dir = os.path.join('saved_matrix',args.dataset, args.method_sim)
	labels_np = np.load(os.path.join(matrix_dir,f'subset_label_{sub_units}.npy'))
	val_labels_np = np.load(os.path.join(matrix_dir, f'subset_val_label_{sub_units}.npy'))
	subset_val_spectrograms = np.load(os.path.join(matrix_dir, f'subset_val_spectrogram_{sub_units}.npy'))
	spectrograms = np.load(os.path.join(matrix_dir,f'subset_spectrogram_{sub_units}.npy'))  
	flattened_spectrograms = flatten_spectrograms(spectrograms)
	flattened_val_spectrograms = flatten_spectrograms(subset_val_spectrograms)
	print(flattened_val_spectrograms.shape)
	# Train and evaluate SVM for spectrogram embeddings
	accuracy_spectrogram = train_evaluate_svm( X_train=flattened_spectrograms, X_test=flattened_val_spectrograms, y_train=labels_np, y_test=val_labels_np)
	logging.info(f'SVM Model Accuracy: {accuracy_spectrogram}')
