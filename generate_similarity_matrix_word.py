import argparse 
import pickle
import numpy as np
import gensim.downloader as api
import eng_to_ipa as ipa
from sklearn.feature_extraction.text import CountVectorizer
import logging
import pandas as pd
import os 

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
def compute_one_hot_representation(words,dataset):
    # Set of all unique letters
    letters = set(''.join(words))
    
    # List of letters sorted for consistent ordering
    letters = sorted(letters)
    
    # Map letters to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
     # Save label_names to a file using pickle
    with open(f'phon_idx_{dataset}.pkl', 'wb') as f:
        pickle.dump(letter_to_index, f)
    # Initialize the one-hot representation matrix
    size = len(letters)
    one_hot_matrix = np.eye(size, dtype=int)
    
    return one_hot_matrix, letters
 
def get_phonetique_from_yemba(xlsx_path):
    df = pd.read_excel(xlsx_path)
    # Clean the phonetique_encoded column
    df['phonetique_encoded'] = df['phonetique_encoded'].apply(lambda x: x.replace('[', '').replace(']', ''))
    yemba_to_phonetique = dict(zip(df['yemba_encoded'], df['phonetique_encoded']))
    return yemba_to_phonetique
 
    
def simi_matrix(method = 'semantics', dataset=None, method_ac='mixed', sub_units=61):


# Load label_names from the file to verify
  # Load label_names from the file to verify
  with open(f'label_names_{dataset}.pkl', 'rb') as f:
    all_label_names = pickle.load(f)
  sub_label_names = np.load(os.path.join('saved_matrix',dataset, method_ac,f'subset_label_{sub_units}.npy'))
  
  label_names = set(all_label_names[sub_label_names])
   #Save label_names to a file using pickle
  with open('subset_label_names_{dataset}.pkl', 'wb') as f:
    pickle.dump(label_names, f)
 
  xlsx_path = f'data/yemba/corpus_words.xlsx' 
  
  if method == 'semantics':
# Load the GloVe Twitter embeddings
    glove_vectors = api.load('glove-twitter-25')


# Retrieve embeddings for each word in the list
    word_embeddings = np.array([glove_vectors[word] for word in label_names])

    similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
  elif method == 'phon_count':
    # Convert words to their phoneme representations
    if dataset=='yemba_command' or dataset=='yemba_command_small':
       yemba_to_phonetique_mapping = get_phonetique_from_yemba(xlsx_path)
       phoneme_words = [yemba_to_phonetique_mapping.get(word) for word in label_names]
    else:
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
    if dataset=='yemba_command':
       yemba_to_phonetique_mapping = get_phonetique_from_yemba(xlsx_path)
       phoneme_words = [yemba_to_phonetique_mapping.get(word) for word in label_names]
    else:
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
    # Convert words to their phoneme representations
    if dataset=='yemba_command':
       yemba_to_phonetique_mapping = get_phonetique_from_yemba(xlsx_path)
       phoneme_words = [yemba_to_phonetique_mapping.get(word) for word in label_names]
    else:
       phoneme_words = [ipa.convert(word) for word in label_names]
    similarity_matrix = compute_edit_distance_matrix(phoneme_words)
    
  elif method == 'phon_coo':
    #glove_vectors = api.load('glove-twitter-25')
    # Convert words to their phoneme representations
    #word_embeddings = np.array([glove_vectors[word] for word in label_names])
    # Convert words to their phoneme representations
    if dataset=='yemba_command':
       yemba_to_phonetique_mapping = get_phonetique_from_yemba(xlsx_path)
       phoneme_words = [yemba_to_phonetique_mapping.get(word) for word in label_names]
    else:
       phoneme_words = [ipa.convert(word) for word in label_names]
    similarity_matrix = compute_edit_distance_matrix(phoneme_words)
    
    similarity_matrix, letters = compute_cooccurrence_matrix(phoneme_words)
   
# Compute the one-hot representation
    word_embeddings, letters = compute_one_hot_representation(phoneme_words, dataset)
    
    
    
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
parser.add_argument('--dataset', help='name of current dataset', required=True)
parser.add_argument('--method_sim_ac', help='', required=True)
parser.add_argument('--sub_units', help='fraction of data', required=True)  

args = parser.parse_args()

similarity_matrix, word_embeddings = simi_matrix(method = args.method,dataset=args.dataset, method_ac = args.method_sim_ac, sub_units =args.sub_units)
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
