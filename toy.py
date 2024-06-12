import numpy as np

# Example list of words
words = ['down' 'go' 'left' 'no' 'right' 'stop' 'up' 'yes']

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
    
    return cooccurrence_matrix, letters

# Function to compute the one-hot representation of letters
def compute_one_hot_representation(words):
    # Set of all unique letters
    letters = set(''.join(words))
    
    # List of letters sorted for consistent ordering
    letters = sorted(letters)
    
    # Map letters to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letters)}
    
    # Initialize the one-hot representation matrix
    size = len(letters)
    one_hot_matrix = np.eye(size, dtype=int)
    
    return one_hot_matrix, letters

# Compute the co-occurrence matrix with default window size of 1
cooccurrence_matrix, letters = compute_cooccurrence_matrix(words)

# Compute the one-hot representation
one_hot_matrix, letters = compute_one_hot_representation(words)

# Print the co-occurrence matrix with letters for better visualization
print("Co-occurrence Matrix:")
print("   ", "  ".join(letters))
for letter, row in zip(letters, cooccurrence_matrix):
    print(f"{letter}:", " ".join(f"{num:2}" for num in row))

# Print the one-hot representation with letters for better visualization
print("\nOne-Hot Representation:")
print("   ", "  ".join(letters))
for letter, row in zip(letters, one_hot_matrix):
    print(f"{letter}:", " ".join(f"{num:2}" for num in row))

