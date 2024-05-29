import argparse 
import pickle
import numpy as np
import gensim.downloader as api










parser = argparse.ArgumentParser()
parser.add_argument('--tw', help='word similarity threshold', required=True)
args = parser.parse_args()









# Load label_names from the file to verify
with open('label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)
    


# Load the GloVe Twitter embeddings
glove_vectors = api.load('glove-twitter-25')


# Retrieve embeddings for each word in the list
word_embeddings = np.array([glove_vectors[word] for word in label_names])

# Compute the cosine similarity matrix
similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
norms = np.linalg.norm(word_embeddings, axis=1)
similarity_matrix = similarity_matrix / norms[:, np.newaxis]
similarity_matrix = similarity_matrix / norms[np.newaxis, :]

# Apply threshold
threshold = float(args.tw)
similarity_matrix[similarity_matrix < threshold] = 0

# Set diagonal to 0 to avoid self-loops
np.fill_diagonal(similarity_matrix, 0)

np.save('filtered_similarity_matrix_word.npy', similarity_matrix)
np.save('word_embedding.npy', word_embeddings )
print("Filtered similarity matrix for word label computed successfully.")
