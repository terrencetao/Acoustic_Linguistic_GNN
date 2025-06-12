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


import panphon
from panphon.segment import Segment
import numpy as np
from itertools import zip_longest




def word_to_articulatory_vectors(word, mode='stack'):
    """
    Convertit un mot anglais en vecteurs articulatoires PanPhon.
    
    Args:
        word (str): mot en anglais (ex: 'banana')
        mode (str): 'stack' pour vecteurs phonème par phonème,
                    'mean' pour un seul vecteur moyen
    
    Returns:
        list of vectors (or 1 vector if mode == 'mean')
    """
    # Initialisation
    ft = panphon.FeatureTable()
    ipa_seq = ipa.convert(word)  # Liste des phonèmes IPA (ex: ['b', 'ə', 'n', 'æ', 'n', 'ə'])
    vectors = []

    
    try:
       vectors = ft.word_to_vector_list(word, numeric=True)
    except IndexError:
       print(f"Phonème non reconnu : {seg}")

    if not vectors:
        return None

    if mode == 'mean':
        return np.mean(vectors, axis=0).tolist()
    elif mode == 'flat':
        return np.array(vectors).ravel()
    else:
        return vectors



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

import random

def split_dict_randomly(data_dict, ratio=0.8, seed=None):
    if not 0 < ratio < 1:
        raise ValueError("Le ratio doit être entre 0 et 1.")
    
    items = list(data_dict.items())
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)
    
    split_index = int(len(items) * ratio)
    return dict(items[:split_index]), dict(items[split_index:])


    
    
def simi_matrix(method = 'semantics', dataset=None, method_ac='mixed', sub_units=61):


# Load label_names from the file to verify

  with open(f'label_names_{dataset}.pkl', 'rb') as f:   # Load all the labels names
    all_labels = pickle.load(f)
    
  label_to_index = {label: index for index, label in enumerate(all_labels)}
  index_to_label = {index: label for index, label in enumerate(all_labels)}
  
  with open(f'label_to_index_{dataset}.pkl', 'wb') as f:
    pickle.dump(label_to_index, f)                      # save the label_to_index which is contains the encode label of each words
    
  with open(f'index_to_label_{dataset}.pkl', 'wb') as f:
    pickle.dump(index_to_label, f)                      #
  
  # random selection of a subset words for induction validation  
  label_dic_names, val_dic_names = split_dict_randomly(label_to_index , ratio=0.8, seed=42)  
    
  with open(f'subset_label_names_{dataset}.pkl', 'wb') as f:
    pickle.dump(label_dic_names, f)
    
  with open(f'induc_val_names_{dataset}.pkl', 'wb') as f:
    pickle.dump(val_dic_names, f)
  
  
  label_names = label_dic_names.keys()
  
 
 
    
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
    
  elif method == 'phon_art':
    word_embeddings = [word_to_articulatory_vectors(word, mode='mean') for word in label_names]
    padded = list(zip_longest(*word_embeddings, fillvalue=0))
    word_embeddings = np.array(padded).T
    similarity_matrix = np.dot(word_embeddings, word_embeddings.T)
    
  elif method == 'phon_coo':

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

np.save(f'filtered_similarity_matrix_word_{args.dataset}.npy', similarity_matrix)
np.save(f'word_embedding_{args.dataset}.npy', word_embeddings )
print("Filtered similarity matrix for word label computed successfully.")
