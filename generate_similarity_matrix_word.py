import argparse 
import pickle
import numpy as np
import gensim.downloader as api
import eng_to_ipa as ipa
from sklearn.feature_extraction.text import CountVectorizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_edit_distance_matrix(words):
    n = len(words)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            dist = levenshtein_distance(words[i], words[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix
    
# Function to compute the co-occurrence matrix of letters with a given window size
def compute_cooccurrence_matrix(words, window=1):
    # Set of all unique letters
    letters = set(''.join(words))
    
    # List of letters sorted for consistent ordering
    letters = sorted(letters)
    
    # Map letters to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
    
    # Initialize the co-occurrence matrix
    size = len(letters)
    cooccurrence_matrix = np.zeros((size, size), dtype=int)
    
    # Populate the co-occurrence matrix
    for word in words:
        for i, letter1 in enumerate(word):
            idx1 = letter_to_index[letter1]
            for j in range(max(0, i - window), min(len(word), i + window + 1)):
                if i != j:
                    letter2 = word[j]
                    idx2 = letter_to_index[letter2]
                    cooccurrence_matrix[idx1, idx2] += 1
    
    return normalize_matrix(cooccurrence_matrix), letters
    
def normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = matrix / row_sums
    normalized_matrix[np.isnan(normalized_matrix)] = 0  # Replace NaN with 0
    return normalized_matrix
    
# Function to compute the one-hot representation of letters
def compute_one_hot_representation(words):
    # Set of all unique letters
    letters = set(''.join(words))
    
    # List of letters sorted for consistent ordering
    letters = sorted(letters)
    
    # Map letters to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
     # Save label_names to a file using pickle
    with open('phon_idx.pkl', 'wb') as f:
        pickle.dump(letter_to_index, f)
    # Initialize the one-hot representation matrix
    size = len(letters)
    one_hot_matrix = np.eye(size, dtype=int)
    
    return one_hot_matrix, letters
 
 
 
    
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
  
  elif method == 'phon_suit':
    glove_vectors = api.load('glove-twitter-25')
    # Convert words to their phoneme representations
    word_embeddings = np.array([glove_vectors[word] for word in label_names])
    phoneme_words = [ipa.convert(word) for word in label_names]
    similarity_matrix = compute_edit_distance_matrix(phoneme_words)
    
  elif method == 'phon_coo':
    similarity_matrix, letters = compute_cooccurrence_matrix(label_names)
   
# Compute the one-hot representation
    word_embeddings, letters = compute_one_hot_representation(label_names)
    
    
    
    # Print the co-occurrence matrix with letters for better visualization
    print("Co-occurrence Matrix:")
    print("   ", "  ".join(letters))
    for letter, row in zip(letters, similarity_matrix):
      print(f"{letter}:", " ".join(f"{num:2}" for num in row))

# Print the one-hot representation with letters for better visualization
    print("\nOne-Hot Representation:")
    print("   ", "  ".join(letters))
    for letter, row in zip(letters,  word_embeddings):
      print(f"{letter}:", " ".join(f"{num:2}" for num in row))
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
