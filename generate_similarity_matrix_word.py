import argparse 
import pickle
import numpy as np
import gensim.downloader as api
import eng_to_ipa as ipa
from sklearn.feature_extraction.text import CountVectorizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simi_matrix(method = 'semantics'):


# Load label_names from the file to verify
  # Load label_names from the file to verify
  with open('label_names.pkl', 'rb') as f:
    all_label_names = pickle.load(f)
  sub_label_names = np.load('subset_label.npy')
  
  label_names = set(all_label_names[sub_label_names])
  # Save label_names to a file using pickle
  with open('subset_label_names.pkl', 'wb') as f:
    pickle.dump(label_names, f)

  if method == 'semantics':
# Load the GloVe Twitter embeddings
    glove_vectors = api.load('glove-twitter-25')


# Retrieve embeddings for each word in the list
    word_embeddings = np.array([glove_vectors[word] for word in label_names])

    similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
  elif method == 'phon_count':
    # Convert words to their phoneme representations
    phoneme_words = [ipa.convert(word) for word in label_names]


# Initialize CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', token_pattern=r'[^ ]')

# Fit the vectorizer on the phoneme words and transform them to vectors
    X = vectorizer.fit_transform(phoneme_words)


# Retrieve embeddings for each word in the list
    word_embeddings = X.toarray()
    similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
  elif method == 'mixed':
    glove_vectors = api.load('glove-twitter-25')


# Retrieve embeddings for each word in the list
    word_embeddings = np.array([glove_vectors[word] for word in label_names])
    # Convert words to their phoneme representations
    phoneme_words = [ipa.convert(word) for word in label_names]


# Initialize CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', token_pattern=r'[^ ]')

# Fit the vectorizer on the phoneme words and transform them to vectors
    X = vectorizer.fit_transform(phoneme_words)
    vectors = X.toarray()
    similarity_matrix = np.dot(vectors, vectors.T)
  # Compute the cosine similarity matrix
  
  return similarity_matrix, word_embeddings



parser = argparse.ArgumentParser()
parser.add_argument('--tw', help='word similarity threshold', required=True)
parser.add_argument('--method', help='method to compute a word similarity', required=True)

args = parser.parse_args()
similarity_matrix, word_embeddings = simi_matrix(method = args.method)
print(similarity_matrix.shape)

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
