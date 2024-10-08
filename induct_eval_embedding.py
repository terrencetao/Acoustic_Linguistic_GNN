import os
import torch
import dgl
import numpy as np
import csv
from gnn_heto_model import HeteroGCN
from gnn_heto_with_attention_model import HeteroGCNWithAllAttention
from gnn_model import GCN
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from generate_similarity_matrix_acoustic import compute_distance_for_pair, compute_dtw_distance, distance_dtw, vgg_distance
from heterogenous_graph import filter_similarity_matrix
import logging
from joblib import Parallel, delayed
import tensorflow as tf
from weak_ML2 import SimpleCNN,evaluate_cnn
from tqdm import tqdm
import math
from weakDense import SimpleDense, evaluate_dense, train_dense
     
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)
    

def ml_distance(ml_model, spec1, spec2):
    ml_model.eval()
    # Ensure the input spectrograms are in the correct shape for the model
    spec1 = torch.tensor(spec1).unsqueeze(0)  # Add batch dimension
    spec2 = torch.tensor(spec2).unsqueeze(0)  # Add batch dimension
    
    # Stack the spectrograms into a batch
    batch = torch.stack([spec1, spec2], dim=0)
    
    
    
    # Pass the batch through the model
    with torch.no_grad():  # Disable gradient calculation
        logits = ml_model(batch)
        ml_predictions = F.softmax(logits, dim=1)
    
    # Calculate the KL divergence (or another distance measure) between the predictions
    kl_divergence = F.kl_div(ml_predictions[0].log(), ml_predictions[1], reduction='batchmean')
    
    return kl_divergence



def add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, distance_function, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    # Randomly select k nodes from the existing graph for each new node
    existing_indices = np.arange(num_existing_nodes)
    
    def process_new_node(new_node_index):
        selected_indices = np.random.choice(existing_indices, k, replace=False)
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        edges = []
        for i in selected_indices:
            distance = distance_function(new_node_spectrogram, dgl_G.ndata['feat'][i].numpy())
            similarity = np.exp(-distance)
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    for edges in all_edges:
        for src, dst, weight in edges:
            dgl_G.add_edges(src, dst, {'weight': torch.tensor([weight], dtype=torch.float32)})
    
    return dgl_G, num_existing_nodes


    
def add_new_nodes_to_graph_randomly2(dgl_G, new_node_spectrograms, k, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    # Randomly select k nodes from the existing graph for each new node
    existing_indices = np.arange(num_existing_nodes)
    
    def process_new_node(new_node_index):
        selected_indices = np.random.choice(existing_indices, k, replace=False)
        
        edges = []
        weights = []
        for i in selected_indices:
            similarity = 1
            edges.append((new_node_index, i))
            edges.append((i, new_node_index))
            weights.append(similarity)
            weights.append(similarity)
        return edges, weights

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_weights = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and weights
    all_edges = [edge for edges, _ in all_edges_and_weights for edge in edges]
    all_weights = [weight for _, weights in all_edges_and_weights for weight in weights]

    # Convert lists to tensors
    src, dst = zip(*all_edges)
    src = torch.tensor(src, dtype=torch.int64)
    dst = torch.tensor(dst, dtype=torch.int64)
    weights = torch.tensor(all_weights, dtype=torch.float32)

    # Add edges and weights to the graph in a single batch operation
    dgl_G.add_edges(src, dst, {'weight': weights})
    
    return dgl_G, num_existing_nodes

def add_new_nodes_to_graph_knn(dgl_G, new_node_spectrograms, k, distance_function, ml, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    existing_features = dgl_G.ndata['feat'][:num_existing_nodes].numpy()

    def process_new_node(new_node_index):
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        # Compute distances to all existing nodes
        distances = [distance_function(new_node_spectrogram, existing_features[i]) for i in range(num_existing_nodes)]
        
        # Select the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        edges = []
        for i in nearest_indices:
            distance = distances[i]
            similarity = np.exp(-distance)
            #similarity = distance
            
            if similarity > 0.5:
               edges.append((new_node_index, i, 1))
               edges.append((i, new_node_index, 1))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    src_nodes, dst_nodes, weights = zip(*[(src, dst, weight) for edges in all_edges for src, dst, weight in edges])
    dgl_G.add_edges(src_nodes, dst_nodes, {'weight': torch.tensor(weights, dtype=torch.float32)})

    return dgl_G, num_existing_nodes
    
    
def add_new_nodes_to_graph_random(dgl_G, new_node_spectrograms, new_node_labels, existing_node_labels, k, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features
    
    # Set labels for new nodes
    new_labels = torch.from_numpy(new_node_labels)
    dgl_G.ndata['label'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_labels
    
    # Existing labels are passed as a parameter instead of extracted from dgl_G
    existing_labels = existing_node_labels

    def process_new_node(new_node_index, new_node_label):
        # Select k random nodes with the same label
        same_label_indices = np.where(existing_labels == new_node_label)[0]
        if len(same_label_indices) < k:
            nearest_indices = same_label_indices
        else:
            nearest_indices = np.random.choice(same_label_indices, size=k, replace=False)
        
        edges = []
        for i in nearest_indices:
            similarity = np.exp(-np.random.rand())  # Use a random similarity for random connection
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index, new_node_labels[new_node_index - num_existing_nodes]) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    src_nodes, dst_nodes, weights = zip(*[(src, dst, weight) for edges in all_edges for src, dst, weight in edges])
    dgl_G.add_edges(src_nodes, dst_nodes, {'weight': torch.tensor(weights, dtype=torch.float32)})

    return dgl_G, num_existing_nodes
    
def add_edges_without_duplicates(graph, src, dst, weights, etype):
    existing_edges = set(zip(graph.edges(etype=etype)[0].numpy(), graph.edges(etype=etype)[1].numpy()))
    
    new_edges = []
    new_weights = []
    for s, d, w in zip(src, dst, weights):
        if (s, d) not in existing_edges:
            new_edges.append((s, d))
            new_weights.append(w)

    if new_edges:
        new_src, new_dst = zip(*new_edges)
        graph.add_edges(new_src, new_dst, {'weight': torch.tensor(new_weights, dtype=torch.float32)}, etype=etype)
    
    
def add_new_acoustic_nodes_to_hetero_graph(hetero_graph, homograph, new_node_spectrograms, k, ml_model, threshold_probability, n_jobs=-1):
    num_existing_acoustic_nodes = hetero_graph.num_nodes('acoustic')
    num_existing_word_nodes = hetero_graph.num_nodes('word')
    num_new_nodes = new_node_spectrograms.shape[0]
    print(num_new_nodes)
    # Transfer the results back to the hetero_graph
    src, dst = homograph.edges()
    weights = homograph.edata['weight']
    
    add_edges_without_duplicates(hetero_graph, src, dst, weights, etype=('acoustic', 'sim_tic','acoustic'))
    # Add features for the new 'acoustic' nodes
    flattened_spectrograms = flatten_spectrograms(new_node_spectrograms)
    new_features = torch.from_numpy(flattened_spectrograms)
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = new_features
    
    
    # Get softmax probabilities for connections to 'word' nodes using the ML model
    ml_model.eval()
    new_node_features = torch.tensor(new_node_spectrograms)
    
# Predict softmax probabilities
    with torch.no_grad():  # Disable gradient calculation
         logits = ml_model(new_node_features)
         ml_predictions = F.softmax(logits, dim=1)
         
    
    ml_probabilities = ml_predictions
    
    # Filter probabilities based on the threshold
    ml_probabilities = filter_similarity_matrix(ml_probabilities.numpy(), threshold=threshold_probability, k=k)
    
    def process_new_node(new_node_index):
        edges = []
        probabilities = []
        for word_node_index in range(num_existing_word_nodes):
            similarity = ml_probabilities[new_node_index - num_existing_acoustic_nodes, word_node_index]
            if similarity > 0:
                edges.append((new_node_index, word_node_index))
                probabilities.append(similarity)
        return edges, probabilities

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_probs = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    all_edges = [edge for edges, _ in all_edges_and_probs for edge in edges]
    all_probabilities = [prob for _, probs in all_edges_and_probs for prob in probs]
    
    if all_edges:
        src, dst = zip(*all_edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        hetero_graph.add_edges(src, dst, etype=('acoustic', 'related_to', 'word'))
        new_weights = torch.tensor(all_probabilities, dtype=torch.float32)
        if 'weight' not in hetero_graph.edges['related_to'].data:
           num_existing_edges = hetero_graph.num_edges(('acoustic', 'related_to', 'word'))
           hetero_graph.edges['related_to'].data['weight'] = torch.zeros(num_existing_edges, dtype=torch.float32)
    
    # Assign new edge weights to the new edges only
        hetero_graph.edges['related_to'].data['weight'][-new_weights.shape[0]:] = new_weights

        #hetero_graph.edges['related_to'].data['weight'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = torch.tensor(all_probabilities, dtype=torch.float32)
    return hetero_graph, num_existing_acoustic_nodes

    
def add_new_nodes_to_hetero_graph_knn(hetero_graph, new_node_spectrograms, k, distance_function, n_jobs=-1):
    # Work with 'acoustic' node type specifically
    num_existing_nodes = hetero_graph.num_nodes('acoustic')
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new 'acoustic' nodes to the graph
    hetero_graph.add_nodes(num_new_nodes, ntype='acoustic')

    # Add features for the new 'acoustic' nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    existing_features = hetero_graph.nodes['acoustic'].data['feat'][:num_existing_nodes].numpy()

    def process_new_node(new_node_index):
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        # Compute distances to all existing 'acoustic' nodes
        distances = [distance_function(new_node_spectrogram, existing_features[i]) for i in range(num_existing_nodes)]
        
        # Select the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        edges = []
        for i in nearest_indices:
            distance = distances[i]
            similarity = np.exp(-distance)
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))

    # Flatten the list of edges and add them to the graph
    for edges in all_edges:
        src, dst, weights = zip(*edges)
        hetero_graph.add_edges(src, dst, {'weight': torch.tensor(weights, dtype=torch.float32)}, etype=('acoustic', 'sim_tic', 'acoustic'))
      
    return hetero_graph,num_existing_nodes
    


def add_new_acoustic_nodes_to_hetero_graph_knn(hetero_graph, new_node_spectrograms, k, distance_function, ml_model, threshold_probability, n_jobs=-1):
    # Step 1: Add new acoustic nodes and connect to existing acoustic nodes using KNN
    hetero_graph, num_existing_acoustic_nodes = add_new_nodes_to_hetero_graph_knn(
        hetero_graph, 
        new_node_spectrograms, 
        k, 
        distance_function, 
        n_jobs
    )
    
    # Step 2: Connect new acoustic nodes to word nodes using the ML model
    num_existing_word_nodes = hetero_graph.num_nodes('word')
    num_new_nodes = new_node_spectrograms.shape[0]
    # Get softmax probabilities for connections to 'word' nodes using the ML model
    ml_model.eval()
    new_node_features = torch.tensor(new_node_spectrograms)
    
# Predict softmax probabilities
    with torch.no_grad():  # Disable gradient calculation
         logits = ml_model(new_node_features)
         ml_predictions = F.softmax(logits, dim=1)
         
    
    ml_probabilities = ml_predictions
    
    # Filter probabilities based on the threshold
    ml_probabilities = filter_similarity_matrix(ml_probabilities.numpy(), threshold=threshold_probability, k=k)
    
    def process_new_node(new_node_index):
        edges = []
        probabilities = []
        for word_node_index in range(num_existing_word_nodes):
            similarity = ml_probabilities[new_node_index - num_existing_acoustic_nodes, word_node_index]
            if similarity > 0:
                edges.append((new_node_index, word_node_index))
                probabilities.append(similarity)
        return edges, probabilities

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_probs = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    all_edges = [edge for edges, _ in all_edges_and_probs for edge in edges]
    all_probabilities = [prob for _, probs in all_edges_and_probs for prob in probs]
    
    if all_edges:
        src, dst = zip(*all_edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        hetero_graph.add_edges(src, dst, etype=('acoustic', 'related_to', 'word'))
        new_weights = torch.tensor(all_probabilities, dtype=torch.float32)
        if 'weight' not in hetero_graph.edges['related_to'].data:
           num_existing_edges = hetero_graph.num_edges(('acoustic', 'related_to', 'word'))
           hetero_graph.edges['related_to'].data['weight'] = torch.zeros(num_existing_edges, dtype=torch.float32)
    
    # Assign new edge weights to the new edges only
        hetero_graph.edges['related_to'].data['weight'][-new_weights.shape[0]:] = new_weights

        #hetero_graph.edges['related_to'].data['weight'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = torch.tensor(all_probabilities, dtype=torch.float32)
    return hetero_graph, num_existing_acoustic_nodes
    
def add_new_acoustic_nodes_to_hetero_graph_random(hetero_graph, new_node_spectrograms, new_node_labels, existing_node_labels, k, n_jobs=-1):
    num_existing_nodes = hetero_graph.num_nodes('acoustic')
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new 'acoustic' nodes to the graph
    hetero_graph.add_nodes(num_new_nodes, ntype='acoustic')

    # Add features for the new 'acoustic' nodes
   
    new_features = torch.from_numpy(new_node_spectrograms).view(new_node_spectrograms.shape[0], -1)
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features
    
    # Set labels for the new nodes
    new_labels = torch.from_numpy(new_node_labels)
    hetero_graph.nodes['acoustic'].data['label'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_labels
    
    # Existing labels are passed as a parameter instead of being extracted from the graph
    existing_labels = existing_node_labels

    def process_new_node(new_node_index, new_node_label):
        # Select k random nodes with the same label
        same_label_indices = np.where(existing_labels == new_node_label)[0]
        if len(same_label_indices) < k:
            selected_indices = same_label_indices
        else:
            selected_indices = np.random.choice(same_label_indices, size=k, replace=False)
        
        edges = []
        for i in selected_indices:
            similarity = np.exp(-np.random.rand())  # Use a random similarity for random connection
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index, new_node_labels[new_node_index - num_existing_nodes]) for new_node_index in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    for edges in all_edges:
        src, dst, weights = zip(*edges)
        hetero_graph.add_edges(src, dst, {'weight': torch.tensor(weights, dtype=torch.float32)}, etype=('acoustic', 'sim_tic', 'acoustic'))

    return hetero_graph, num_existing_nodes

def add_new_links_nodes_to_hetero_graph_random(hetero_graph, new_node_spectrograms, new_node_labels, existing_node_labels, k, ml_model, threshold_probability, n_jobs=-1):
    # Step 1: Add new acoustic nodes and connect to existing acoustic nodes using random label
    hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph_random(
                              hetero_graph, 
                              new_node_spectrograms, 
                              new_node_labels, 
                              existing_node_labels, 
                              k)

    # Step 2: Connect new acoustic nodes to word nodes using the ML model
    num_existing_word_nodes = hetero_graph.num_nodes('word')
    num_new_nodes = new_node_spectrograms.shape[0]
    # Get softmax probabilities for connections to 'word' nodes using the ML model
    ml_model.eval()
    new_node_features = torch.tensor(new_node_spectrograms)
    new_node_features = new_node_features.view(new_node_features.shape[0],1 , new_node_features.shape[1], new_node_features.shape[2])
# Predict softmax probabilities
    with torch.no_grad():  # Disable gradient calculation
         logits = ml_model(new_node_features)
         ml_predictions = F.softmax(logits, dim=1)
         
    
    ml_probabilities = ml_predictions
    
    # Filter probabilities based on the threshold
    ml_probabilities = filter_similarity_matrix(ml_probabilities.numpy(), threshold=threshold_probability, k=k)
    
    def process_new_node(new_node_index):
        edges = []
        probabilities = []
        for word_node_index in range(num_existing_word_nodes):
            similarity = ml_probabilities[new_node_index - num_existing_acoustic_nodes, word_node_index]
            if similarity > 0:
                edges.append((new_node_index, word_node_index))
                probabilities.append(similarity)
        return edges, probabilities

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_probs = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in tqdm(range(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)))
    
    # Flatten the list of edges and add them to the graph
    all_edges = [edge for edges, _ in all_edges_and_probs for edge in edges]
    all_probabilities = [prob for _, probs in all_edges_and_probs for prob in probs]
    
    if all_edges:
        src, dst = zip(*all_edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        hetero_graph.add_edges(src, dst, etype=('acoustic', 'related_to', 'word'))
        new_weights = torch.tensor(all_probabilities, dtype=torch.float32)
        if 'weight' not in hetero_graph.edges['related_to'].data:
           num_existing_edges = hetero_graph.num_edges(('acoustic', 'related_to', 'word'))
           hetero_graph.edges['related_to'].data['weight'] = torch.zeros(num_existing_edges, dtype=torch.float32)
    
    # Assign new edge weights to the new edges only
        hetero_graph.edges['related_to'].data['weight'][-new_weights.shape[0]:] = new_weights

        #hetero_graph.edges['related_to'].data['weight'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = torch.tensor(all_probabilities, dtype=torch.float32)
    return hetero_graph, num_existing_acoustic_nodes
    
      
    
def generate_embeddings(gcn_model, dgl_G,num_existing_nodes, new_node_spectrograms):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    dgl_G (dgl.DGLGraph): The graph structure containing existing nodes.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    
    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new nodes to the graph and get the updated graph and the number of existing nodes
    #dgl_G, num_existing_nodes = add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, compute_distance)
    
    # Extract edge weights and features from the graph
    edge_weights = dgl_G.edata['weight']
    features = dgl_G.ndata['feat']
    
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        embeddings = gcn_model(dgl_G, features, edge_weights).numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_nodes, num_existing_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    # Debugging prints
    print(embeddings.shape)
    print(num_existing_nodes)
    
    return embeddings[:num_existing_nodes], new_node_embeddings


def generate_embeddings_hetero(gcn_model, hetero_graph,num_existing_acoustic_nodes, new_node_spectrograms):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    hetero_graph (dgl.DGLHeteroGraph): The heterogeneous graph structure.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    ml_model (tf.keras.Model): The ML model for predicting connection probabilities.
    threshold_probability (float): The threshold for filtering connection probabilities.
    n_jobs (int): The number of jobs for parallel processing.

    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new 'acoustic' nodes to the graph
    #hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph(
    #    hetero_graph, 
    #    new_node_spectrograms, 
    #    k, 
    #    compute_distance, 
    #    ml_model, 
    #    threshold_probability, 
    #    n_jobs
    #)
    
    # Extract features from the graph
    features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        # Assuming the GCN model takes the graph and node features as input
        embeddings = gcn_model(hetero_graph, features_dic)
        embeddings = embeddings['acoustic'].numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    return embeddings[:num_existing_acoustic_nodes], new_node_embeddings
    
# Function to split data into train and test sets
def train_test_split_data(embeddings, labels, test_size=0.2, random_state=42):
    return train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)

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
    
    
logging.info(f' ----------------------------------------------------- Evaluation of Representation on validate set  -----------------------------------------------')
       
parser = argparse.ArgumentParser()
parser.add_argument('--twa', help='word similarity threshold', required=True)
parser.add_argument('--num_n_h', help='method to compute a word similarity', required=True)
parser.add_argument('--mhg', help='method to compute a word similarity', required=True)
parser.add_argument('--num_n_a', help='method to compute a word similarity', required=True)
parser.add_argument('--ta', help='method to compute a word similarity', required=True)
parser.add_argument('--alpha', help='method to compute a word similarity', required=True)
parser.add_argument('--tw', help='method to compute a word similarity', required=True)
parser.add_argument('--msw', help='method to compute a word similarity', required=True)
parser.add_argument('--msa', help='method to compute heterogeneous similarity', required=True)
parser.add_argument('--mgw', help='method to build word graph ', required=True)
parser.add_argument('--mma', help='method to build acoustic matrix', required=True)
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
parser.add_argument('--sub_units', help='fraction of data', required=True)  
parser.add_argument('--dataset', help='name of dataset', required=True)
args = parser.parse_args()


# Paths
graph_folder = os.path.join('saved_graphs',args.dataset,args.mma,args.msa)
model_folder = 'models'
matrix_folder = os.path.join('saved_matrix',args.dataset, args.mma)
# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder,f"kws_graph_{args.num_n_a}_{args.sub_units}.dgl"))
dgl_G = glist[0]

features = dgl_G.ndata['feat']
print(features.shape)
labels = dgl_G.ndata['label']
subset_val_labels = np.load(os.path.join(matrix_folder,f'subset_val_label_{args.sub_units}.npy'))
subset_val_spectrograms = np.load(os.path.join(matrix_folder,f'subset_val_spectrogram_{args.sub_units}.npy'))

# Define the input features size
#in_feats = features[0].shape[0] * features[0].shape[1]
in_feats = features[0].shape[0]
hidden_size = 64
num_classes = len(torch.unique(labels))
conv_param = [(1, 3, (20, 64)), 32, 2]
hidden_units = [32, 32]
hidden_units = [32, 32]

# Load supervised GCN model
logging.info(f'Load supervised GCN model')
model_sup_path = os.path.join(model_folder, "gnn_model.pth")
loaded_model_sup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_sup.load_state_dict(torch.load(model_sup_path))

kws_graph_path_val = os.path.join(graph_folder, f"kws_graph_val_{args.num_n_a}_{args.sub_units}.dgl")

# Extract labels for training
labels_np = labels.numpy()
val_labels_np = subset_val_labels
acoustic_model = torch.load('models/dense.pth')
if not os.path.isfile(kws_graph_path_val):
  logging.info(f'Extract acoustic node representations from supervised GCN')
  dgl_G, num_existing_nodes = add_new_nodes_to_graph_knn(dgl_G, new_node_spectrograms=subset_val_spectrograms,  k=math.floor(2*int(args.num_n_a)), distance_function=vgg_distance, ml=acoustic_model)
  #dgl_G, num_existing_nodes = add_new_nodes_to_graph_random(dgl_G, new_node_spectrograms=subset_val_spectrograms, new_node_labels=val_labels_np, existing_node_labels=labels_np, k=math.floor(int(args.num_n_a)/4), n_jobs=-1)
  #dgl_G, num_existing_nodes= add_new_nodes_to_graph_randomly2(dgl_G, new_node_spectrograms=subset_val_spectrograms, k=1500, n_jobs=-1)
  print(dgl_G.number_of_nodes())
  kws_graph_path_val = os.path.join(graph_folder, f"kws_graph_val_{args.num_n_a}_{args.sub_units}.dgl")
  dgl.save_graphs(kws_graph_path_val, [dgl_G])
  print(f"dgl val save successfully")
else:
  print(f"File {kws_graph_path_val} already exists. Skipping computation.")
  num_existing_nodes = dgl_G.number_of_nodes()
  glist, label_dict= load_graphs(kws_graph_path_val)
  dgl_G = glist[0]
#print(dgl_G.number_of_nodes())
node_embeddings_sup, node_val_embeddings_sup = generate_embeddings(gcn_model=loaded_model_sup, 
                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, 
                                               )

#print(node_embeddings_sup.shape)
#print(node_val_embeddings_sup.shape)

# Load unsupervised GCN model
logging.info(f'Load unsupervised GCN model')
model_unsup_path = os.path.join(model_folder, "gnn_model_unsup.pth")
loaded_model_unsup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_unsup.load_state_dict(torch.load(model_unsup_path))


# Load unsupervised sage GCN model
logging.info(f'Load unsupervised sage GCN model')
#model_unsup_path_sage = os.path.join(model_folder, "gnn_model_unsup_sage.pth")
#loaded_model_unsup_sage = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
#loaded_model_unsup_sage.load_state_dict(torch.load(model_unsup_path_sage))


# Load the heterogeneous graph
glists, _ = dgl.load_graphs(os.path.join(graph_folder, args.mhg, f"hetero_graph_{args.num_n_h}_{args.sub_units}.dgl"))
hetero_graph = glists[0]

# Load the heterogeneous GCN model
features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
in_feats = {'acoustic': features_dic['acoustic'].shape[1], 'word': features_dic['word'].shape[1]}
hidden_size = 64
out_feats = 16

# Initialize the model
model = HeteroGCN(in_feats, hidden_size, out_feats)
#model_sage = HeteroGCN(in_feats, hidden_size, out_feats)
# Load the pre-trained model state
model.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model.pth")))
model.eval()

#model_sage.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model_unsupervised.pth")))
#model_sage.eval()
# Extract acoustic node representations
logging.info(f'Extract acoustic node representations from hetero GCN')

hetero_graph_path_val = os.path.join(graph_folder, args.mhg, f"hetero_graph_val_{args.mhg}_{args.num_n_h}_{args.sub_units}.dgl")
if not os.path.isfile(hetero_graph_path_val):
# Add new 'acoustic' nodes to the graph
    #hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph(
    #    hetero_graph, dgl_G,
    #    new_node_spectrograms=subset_val_spectrograms, 
    #    k=int(args.num_n_h), 
    #    ml_model=acoustic_model, 
    #    threshold_probability=float(args.twa), 
    #    n_jobs=-1
    #)
    #hetero_graph, num_existing_acoustic_nodes = add_new_links_nodes_to_hetero_graph_random(
    #                    hetero_graph, 
    #                    new_node_spectrograms=subset_val_spectrograms, 
    #                    new_node_labels=val_labels_np, 
    #                    existing_node_labels=labels_np, 
    #                   k=int(args.num_n_h), 
    #                    ml_model=acoustic_model, 
    #                    threshold_probability=float(args.twa), n_jobs=-1)
    hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph_knn(
                                hetero_graph, 
                                 new_node_spectrograms=subset_val_spectrograms, 
                                  k=int(args.num_n_h),
                                distance_function= vgg_distance, 
                                 ml_model=acoustic_model, 
                                  threshold_probability=float(args.twa), 
                                  n_jobs=-1)
    
    
    dgl.save_graphs(hetero_graph_path_val, hetero_graph)
else:
    print(f"File {hetero_graph_path_val} already exists. Skipping computation.")
    num_existing_acoustic_nodes =  hetero_graph.num_nodes('acoustic')
    glist, label_dict = load_graphs(hetero_graph_path_val)
    hetero_graph = glist[0]
acoustic_embeddings, acoustic_val_embeddings = generate_embeddings_hetero(gcn_model=model, 
                                                hetero_graph=hetero_graph,num_existing_acoustic_nodes=num_existing_acoustic_nodes, new_node_spectrograms=subset_val_spectrograms 
                                                )
#acoustic_embeddings_sage, acoustic_val_embeddings_sage = generate_embeddings_hetero(gcn_model=model_sage, 
#                                                hetero_graph=hetero_graph,num_existing_acoustic_nodes=num_existing_acoustic_nodes, new_node_spectrograms=subset_val_spectrograms 
#                                                )




num_heads = 4
logging.info(f'Load unsupervised GCN attention model')
#model_attention_path = os.path.join(model_folder, "hetero_gcn_with_attention_model.pth")
#model_attention = HeteroGCNWithAllAttention(in_feats, hidden_size, out_feats, num_heads=num_heads)
#model_attention.load_state_dict(torch.load(model_attention_path))
#model_attention.eval()

# Extract acoustic node representations
logging.info(f'Extract acoustic node representations')

    
#acoustic_embeddings_attention, acoustic_val_embeddings_attention = generate_embeddings_hetero(gcn_model=model_attention, 
#                                                hetero_graph=hetero_graph, new_node_spectrograms=subset_val_spectrograms)
   


os.makedirs('accuracy', exist_ok=True)
#dataset
dataset = args.dataset
os.makedirs(f'accuracy/{dataset}', exist_ok=True)

# CSV file path
csv_file = f'{dataset}/accuracy_val_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Supervised Model', 'Unsupervised Model','Unsupervised sage Model', 'Heterogeneous Model','Heterogeneous sage Model','Heterogeneous attention Model', 'Spectrogram Baseline', 'CNN Model', 'twa', 'num_n_h', 'mhg', 'num_n_a', 'ta', 'alpha', 'tw', 'msw', 'msa', 'mgw', 'mma'])



# Train and evaluate SVM for supervised embeddings
logging.info(f'Train and evaluate SVM for supervised embeddings')
accuracy_sup = train_evaluate_svm( X_train=node_embeddings_sup, X_test=node_val_embeddings_sup, y_train=labels_np, y_test=val_labels_np)
 
# Train and evaluate SVM for unsupervised embeddings
logging.info(f'Train and evaluate SVM for unsupervised embeddings')
node_embeddings_unsup, node_val_embeddings_unsup = generate_embeddings(gcn_model=loaded_model_unsup, 
                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, )

accuracy_unsup = train_evaluate_svm(X_train=node_embeddings_unsup, X_test=node_val_embeddings_unsup, y_train=labels_np, y_test=val_labels_np)
logging.info(f"Accuracy of unsupervised Model: {accuracy_unsup:.4f}")

# Train and evaluate SVM for unsupervised sage embeddings
logging.info(f'Train and evaluate SVM for unsupervised sage embeddings')
#node_embeddings_unsup_sage, node_val_embeddings_unsup_sage = generate_embeddings(gcn_model=loaded_model_unsup_sage, 
#                                                dgl_G=dgl_G,num_existing_nodes=num_existing_nodes, new_node_spectrograms=subset_val_spectrograms, )

#accuracy_unsup_sage = train_evaluate_svm(X_train=node_embeddings_unsup_sage, X_test=node_val_embeddings_unsup_sage, y_train=labels_np, y_test=val_labels_np)
accuracy_unsup_sage=0.0
logging.info(f"Accuracy of unsupervised sage Model: {accuracy_unsup_sage:.4f}")

# Train and evaluate SVM for heterogeneous model embeddings
logging.info(f'Train and evaluate SVM for heterogeneous model embeddings')

accuracy_hetero = train_evaluate_svm(X_train=acoustic_embeddings, X_test=acoustic_val_embeddings, y_train=labels_np, y_test=val_labels_np)
logging.info(f"Accuracy of the Heterogeneous Model: {accuracy_hetero:.4f}")

#accuracy_hetero_sage = train_evaluate_svm(X_train=acoustic_embeddings_sage, X_test=acoustic_val_embeddings_sage, y_train=labels_np, y_test=val_labels_np)
accuracy_hetero_sage =0.0
logging.info(f"Accuracy of the Heterogeneous sage Model: {accuracy_hetero_sage:.4f}")

# Train and evaluate SVM on the new heterogeneous attention model embeddings
logging.info(f'Train and evaluate SVM on the new heterogeneous attention model embeddings')
#accuracy_attention = train_evaluate_svm(acoustic_embeddings_attention.numpy(), labels_np)
accuracy_attention =0.0
logging.info(f"Accuracy of the Heterogeneous Attention Model: {accuracy_attention:.4f}")



logging.info(f'Train and evaluate SVM for spectrogram embeddings')

spectrograms = np.load(os.path.join(matrix_folder ,f'subset_spectrogram_{args.sub_units}.npy'))
flattened_spectrograms = flatten_spectrograms(spectrograms)
flattened_val_spectrograms = flatten_spectrograms(subset_val_spectrograms)
# Train and evaluate SVM for spectrogram embeddings
accuracy_spectrogram = train_evaluate_svm( X_train=flattened_spectrograms, X_test=flattened_val_spectrograms, y_train=labels_np, y_test=val_labels_np)
logging.info(f'SVM Model Accuracy: {accuracy_spectrogram}')
# Prepare data for CNN

val_spectrograms_tensor = torch.tensor(subset_val_spectrograms, dtype=torch.float32)

val_labels_tensor = torch.tensor(val_labels_np, dtype=torch.long)

val_spectrograms_tensor = val_spectrograms_tensor.unsqueeze(1)
#X_train, X_test, y_train, y_test = train_test_split(spectrograms_tensor, labels_tensor, test_size=0.2, random_state=42)
X_test, y_test =  val_spectrograms_tensor, val_labels_tensor

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define and train the CNN model
logging.info(f'train the CNN model')
input_shape = val_spectrograms_tensor.shape[1:]  # (1, height, width)
cnn_model = torch.load('models/dense.pth')
accuracy_cnn = evaluate_dense(cnn_model, test_loader)
logging.info(f'CNN Model Accuracy: {accuracy_cnn}')

# Write accuracy results to CSV file
logging.info(f'Write accuracy results to CSV file')
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy_sup, accuracy_unsup, accuracy_unsup_sage, accuracy_hetero, accuracy_hetero_sage, accuracy_attention,accuracy_spectrogram, accuracy_cnn, float(args.twa), float(args.num_n_h), args.mhg, float(args.num_n_a), float(args.ta), float(args.alpha), float(args.tw), args.msw, args.msa, args.mgw, args.mma])

