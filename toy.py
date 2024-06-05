import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def random_dtw(matrix, k, spectrogram, alpha, distance_function, n_jobs=-1):
    # Ensure randomly_select_k is properly defined
    random_matrix = randomly_select_k(matrix, k, alpha)
    n = matrix.shape[0]
    
    def process_row(i):
        valid_indices = np.where(matrix[i, :] == 0)[0]
        if len(valid_indices) > 0:
            k_actual = min(k, len(valid_indices))  # Ensure no replacement if not enough valid indices
            selected_indices = np.random.choice(valid_indices, k_actual, replace=False)
            distances = np.array([distance_function(spectrogram, i, j)[2] for j in selected_indices])
            sorted_indices = selected_indices[np.argsort(distances)[:k_actual]]
            sorted_distances = np.sort(distances)[:k_actual]
            return i, sorted_indices, sorted_distances
        else:
            return i, np.array([]), np.array([])
    
    with Parallel(n_jobs=n_jobs) as parallel:
        results = list(tqdm(parallel(delayed(process_row)(i) for i in range(n)), total=n))
    random_matrix[0,[2,1]] = np.array([10,10])
    for i, nearest_indices, distances in results:
        if len(nearest_indices) > 0:
            print(f"Row: {i}, Nearest Indices: {nearest_indices}, Distances: {distances}")
            exp_distances = np.exp(-distances)
            print(f"Exp(-distances): {exp_distances}")
            random_matrix[i, nearest_indices] = exp_distances
            print(f"Updated random_matrix[{i}, {nearest_indices}]: {random_matrix[i, nearest_indices]}")
    
    np.fill_diagonal(random_matrix, 0)
    return random_matrix

# Example usage
# Ensure you have these functions and variables defined:
# matrix, k, spectrogram, alpha, distance_function

# Define a dummy `randomly_select_k` for demonstration purposes
def randomly_select_k(matrix, k, alpha=1):
    new_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        ones_indices = np.where(matrix[i] != 0)[0]
        if len(ones_indices) > k:
            selected_indices = np.random.choice(ones_indices, k, replace=False)
        else:
            selected_indices = ones_indices
        new_matrix[i, selected_indices] = alpha*matrix[i, selected_indices]

    np.fill_diagonal(new_matrix, 0)
    return new_matrix

# Define a dummy distance function for demonstration purposes
def dummy_distance_function(spectrogram, i, j):
    return (i, j, np.random.rand())

# Test with dummy data
matrix = np.zeros((5, 5))
k = 2
spectrogram = np.random.rand(5, 10)
alpha = 0.5

random_matrix = random_dtw(matrix, k, spectrogram, alpha, dummy_distance_function)
print("Final random_matrix:")
print(random_matrix)

