import numpy as np
import networkx as nx
import dgl
import pickle
import torch
import os
import argparse
import netbone as nb
from netbone.filters import boolean_filter, threshold_filter, fraction_filter
from generate_similarity_matrix_acoustic import compute_distance_for_pair, compute_dtw_distance, distance_dtw, vgg_distance
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)
    
def filter_similarity_matrix(similarity_matrix, labels, threshold=0, alpha=1, k=None):
    filtered_matrix = similarity_matrix.copy()
    n = similarity_matrix.shape[0]
    
    for i in range(n):
        valid_indices = np.where(similarity_matrix[i, :] > threshold)[0]
        
        if k is not None and len(valid_indices) > k:
            sorted_indices = valid_indices[np.argsort(similarity_matrix[i, valid_indices])[-k:]]
        else:
            sorted_indices = valid_indices
        
        for j in range(n):
            if i == j or j not in sorted_indices:
                filtered_matrix[i, j] = 0
            elif labels[i] == labels[j]:
                filtered_matrix[i, j] = alpha * similarity_matrix[i, j]

    np.fill_diagonal(filtered_matrix, 0)
    return filtered_matrix






def randomly_select_k(matrix, k, alpha=1):
    new_matrix = np.zeros_like(matrix)
    matrix_size = matrix.shape[0]
    all_ones_indices = []

    for i in range(matrix_size):
        ones_indices = np.where(matrix[i, i+1:] != 0)[0] + (i + 1)  # Only consider upper triangle
        if len(ones_indices) > k:
            selected_indices = np.random.choice(ones_indices, k, replace=False)
        else:
            selected_indices = ones_indices
        
        all_ones_indices.append(ones_indices)
        new_matrix[i, selected_indices] = alpha * matrix[i, selected_indices]
        new_matrix[selected_indices, i] = alpha * matrix[selected_indices, i]  # Ensure symmetry

    np.fill_diagonal(new_matrix, 0)
    
    
    
    return new_matrix, all_ones_indices

    
    
def random_dtw(matrix, k, spectrogram, alpha, distance_function, n_jobs=-1):
    """
    Perform random Dynamic Time Warping (DTW) distance calculations on a given matrix.
    
    Parameters:
    matrix (np.ndarray): Input matrix of size (n, n) where n is the number of nodes.
    k (int): Number of nearest neighbors to consider.
    spectrogram (np.ndarray): Spectrogram data for distance calculation.
    alpha (float): Parameter for the randomly_select_k function.
    distance_function (function): Custom distance function that takes (spectrogram, i, j) as arguments and returns a distance.
    n_jobs (int): Number of jobs for parallel processing. Default is -1 (use all processors).
    
    Returns:
    np.ndarray: Updated matrix with calculated distances.
    """
    
    # Ensure randomly_select_k is properly defined
    random_matrix, ones = randomly_select_k(matrix, k, alpha)
    random_matrix = random_matrix.astype(np.float64)
    n = matrix.shape[0]
    
    def process_row(i,k):
        # Only consider the upper triangle
        valid_indices = np.where(matrix[i, i+1:] == 0)[0] + (i + 1)
        valid_indices = [idx for idx in valid_indices if idx not in ones[i]]
        
        if len(valid_indices) > 0:
            k_actual = min(k, len(valid_indices))  # Ensure no replacement if not enough valid indices
            selected_indices = np.random.choice(valid_indices, k_actual, replace=False)  # You can randomly select k_actual if needed
            distances = np.array([distance_function(spectrogram, i, j)[2] for j in selected_indices])
            sorted_indices = selected_indices[np.argsort(distances)[:k_actual]]
            sorted_distances = np.sort(distances)[:k_actual]
            return i, sorted_indices, sorted_distances
        else:
            return i, np.array([]), np.array([])
    
    with Parallel(n_jobs=n_jobs) as parallel:
        results = list(parallel(delayed(process_row)(i,1) for i in tqdm(range(n))))
    
    for i, nearest_indices, distances in results:
        if len(nearest_indices) > 0:
            exp_distances = np.exp(-distances)
            random_matrix[i, nearest_indices] = exp_distances
    
    # Mirror the upper triangle into the lower triangle for symmetry
    lower_indices = np.tril_indices(n, -1)
    random_matrix[lower_indices] = random_matrix.T[lower_indices]
    
    np.fill_diagonal(random_matrix, 0)  # Ensure diagonal is 0
    return random_matrix


def k_nearest_neighbors(similarity_matrix, k_in, spectrogram, distance_function, k_out=1,n_jobs=-1):
    """
    Compute k-nearest neighbors matrix based on similarity matrix and distance function.
    
    For similar pairs (similarity_matrix[i,j] == 1), keeps the k_in nearest neighbors.
    For dissimilar pairs (similarity_matrix[i,j] == 0), keeps the k_out most distant neighbors.
    
    Args:
        similarity_matrix: Square matrix where 1 indicates similarity, 0 indicates dissimilarity
        k_in: Number of nearest neighbors to keep for similar pairs
        spectrogram: Input data used for distance calculation
        distance_function: Function that computes distance between two items in spectrogram
        n_jobs: Number of parallel jobs
        k_out: Number of most distant neighbors to keep for dissimilar pairs
        
    Returns:
        knn_matrix: Symmetric matrix where:
            - 1 indicates a kept similar pair
            - -1 indicates a kept dissimilar pair
            - 0 indicates no relation
    """
    n = similarity_matrix.shape[0]
    knn_matrix = np.zeros_like(similarity_matrix)
    
    def process_similar(i):
        """Process similar items (where similarity_matrix[i,j] == 1)"""
        # Get indices of similar items above the diagonal
        valid_indices = np.where(similarity_matrix[i, i+1:] == 1)[0] + (i + 1)
        if len(valid_indices) > 0:
            # Calculate distances to these similar items
            distances = np.array([distance_function(spectrogram, i, j)[2] for j in valid_indices])
            # Get k_in nearest similar items (smallest distances)
            nearest_indices = valid_indices[np.argsort(distances)[:k_in]]
            return i, nearest_indices
        return i, np.array([])
    
    def process_dissimilar(i):
        """Process dissimilar items (where similarity_matrix[i,j] == 0)"""
        # Get indices of dissimilar items above the diagonal
        valid_indices = np.where(similarity_matrix[i, i+1:] == 0)[0] + (i + 1)
        if len(valid_indices) > 0:
            # Calculate distances to these dissimilar items
            distances = np.array([distance_function(spectrogram, i, j)[2] for j in valid_indices])
            # Get k_out most dissimilar items (largest distances)
            # Using argsort()[::-1] to sort in descending order
            farthest_indices = valid_indices[np.argsort(distances)[-k_out:][::-1]]
            return i, farthest_indices
        return i, np.array([])
    
    # Process similar items in parallel
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(process_similar)(i) for i in tqdm(range(n), desc="Processing similar items"))
        for i, nearest_indices in results:
            if len(nearest_indices)>0:
               knn_matrix[i, nearest_indices] = 1
    
    # Process dissimilar items in parallel
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(process_dissimilar)(i) for i in tqdm(range(n), desc="Processing dissimilar items"))
        for i, farthest_indices in results:
            if len(farthest_indices)>0:
               knn_matrix[i, farthest_indices] = -1
    
    # Make matrix symmetric by copying upper triangle to lower triangle
    knn_matrix = np.triu(knn_matrix) + np.triu(knn_matrix, 1).T
    np.fill_diagonal(knn_matrix, 0)
    
    return knn_matrix

    
def filtered_matrix(method, subset_labels, similarity_matrix,spectrogram, threshold=None, alpha=None, k=None,k_out=None, distance_function=None):
    filtered_similarity_matrix = np.zeros_like(similarity_matrix)
    if method == 'dtw': 
        filtered_similarity_matrix = filter_similarity_matrix(similarity_matrix, subset_labels, threshold, alpha, k)
    elif method == 'fixed':
        filtered_similarity_matrix = randomly_select_k(similarity_matrix, k)
    elif method == 'mixed':
        if distance_function is None:
            raise ValueError("Distance function must be provided for 'mixed' method")
        filtered_similarity_matrix = random_dtw(similarity_matrix, k, spectrogram,alpha, distance_function)
    elif method == 'knn':
        if distance_function is None:
            raise ValueError("Distance function must be provided for 'knn' method")
        filtered_similarity_matrix = k_nearest_neighbors(similarity_matrix=similarity_matrix,k_in = k, spectrogram = spectrogram, distance_function = distance_function, k_out=k_out)
    elif method == 'filter':
        filtered_similarity_matrix = similarity_matrix
    else:
        raise ValueError("Unsupported method: choose from 'dtw', 'fixed', 'mixed', or 'knn'")

    return filtered_similarity_matrix

def build_dgl_graph(nx_graph):
    # Extraire les arêtes avec poids
    edges = np.array(list(nx_graph.edges(data='weight', default=1.0)), dtype=object)

    if edges.shape[0] == 0:
        # Cas d'un graphe vide
        dgl_graph = dgl.graph(([], []), num_nodes=nx_graph.number_of_nodes())
        dgl_graph.edata['weight'] = torch.tensor([], dtype=torch.float32)
        return dgl_graph

    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    weights = edges[:, 2].astype(float)

    # Création du graphe DGL
    dgl_graph = dgl.graph((src, dst), num_nodes=nx_graph.number_of_nodes())
    dgl_graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return dgl_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_n', help='number of neighbors for filtering acoustic graph', type=int, required=True)
    parser.add_argument('--k_out', help='number of negative  neighbors for filtering acoustic graph', type=int, required=True)
    parser.add_argument('--ta', help='acoustic similarity threshold', type=float, default=0)
    parser.add_argument('--alpha', help='coefficient', type=float, default=2)
    parser.add_argument('--method', help='method for filtering', choices=['dtw', 'fixed', 'mixed', 'knn', 'filter'], required=True)
    parser.add_argument('--dataset', help='name of dataset', required=True)
    parser.add_argument('--sub_units', help='fraction of data', required=True) 
    parser.add_argument('--method_sim', help='', required=True)
    args = parser.parse_args()
    sub_units = int(args.sub_units)
    
    matrix_dir = os.path.join('saved_matrix',args.dataset, args.method_sim)
    matrix_with_labels = np.load(os.path.join(matrix_dir,f'similarity_matrix_with_labels_{sub_units}.npy'))
    labels = matrix_with_labels[:, 0]
    similarity_matrix = matrix_with_labels[:, 1:]
    subset_labels = np.load(os.path.join(matrix_dir,f'subset_label_{sub_units}.npy'))
    subset_spectrogram = np.load(os.path.join(matrix_dir,f'subset_spectrogram_{sub_units}.npy'))

    #print(subset_spectrogram[0])
    save_dir = os.path.join('saved_graphs',args.dataset,args.method_sim, args.method)
    os.makedirs(save_dir, exist_ok=True)
    kws_graph_path = os.path.join(save_dir, f"kws_graph_{args.num_n}_{args.k_out}_{sub_units}.dgl")
    if not os.path.isfile(kws_graph_path):
        filtered_similarity_matrix = filtered_matrix(
        method=args.method,
        subset_labels=subset_labels,
        similarity_matrix=similarity_matrix,
        threshold=args.ta,
        alpha=args.alpha,
        k=int(args.num_n),
        k_out = int(args.k_out),
        spectrogram = subset_spectrogram,
        distance_function=compute_distance_for_pair if args.method == 'knn' or args.method == 'mixed'  else None
    )

        f_matrix_with_labels = np.hstack((subset_labels[:, np.newaxis], filtered_similarity_matrix))
    #print(filtered_similarity_matrix)
        print("Filtered similarity matrix computed successfully.")
        np.save(os.path.join(matrix_dir,f'filtered_matrix_with_labels_{args.num_n}_{args.k_out}_{sub_units}.npy'), f_matrix_with_labels)
       
        G = nx.Graph()
        num_nodes = filtered_similarity_matrix.shape[0]
        G.add_nodes_from(range(num_nodes))

        for i in range(num_nodes):
           for j in range(i + 1, num_nodes):
              similarity = filtered_similarity_matrix[i, j]
              #if similarity > 0:
              G.add_edge(i, j, weight=similarity)
        ## filtrage netbone 
        # 1. Calculer le squelette à haute saillance
        backbone = nb.high_salience_skeleton(G)

        backbone_G = threshold_filter(backbone, 0.001)
        backbone_df = backbone.to_dataframe()
        print(backbone_df.head())
        # Créer un graphe avec tous les nœuds d'origine
        backbone_G_full = nx.Graph()
        backbone_G_full.add_nodes_from(G.nodes())

        # Ajouter ensuite les arêtes filtrées
        backbone_G_full.add_edges_from(backbone_G.edges(data=True))


        #backbone_G = nx.Graph(backbone_G)
        #dgl_G = dgl.from_networkx(backbone_G)
        dgl_G = build_dgl_graph(backbone_G_full)
        dgl_G.ndata['label'] = torch.tensor(labels, dtype=torch.long)
        dgl_G.ndata['feat'] = torch.stack([torch.from_numpy(spec) for spec in subset_spectrogram])

        print("Number of nodes:", dgl_G.number_of_nodes())
        print("Number of edges:", dgl_G.number_of_edges())

        
        dgl.save_graphs(kws_graph_path, [dgl_G])
    else:
        print(f"File {kws_graph_path} already exists. Skipping computation.")

