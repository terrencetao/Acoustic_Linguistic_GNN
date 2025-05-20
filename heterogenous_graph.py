import dgl
from dgl.data.utils import load_graphs
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import numpy as np
import random
import logging
import pickle
import math
import torch.nn.functional as F
from weak_ML2 import SimpleCNN
from weakDense import SimpleDense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
random.seed(42)
np.random.seed(42)

def filter_similarity_matrix(similarity_matrix,  threshold=0, k=None):
    # Make a copy of the similarity matrix to avoid modifying the original
    filtered_matrix = similarity_matrix.copy()
    
    # Get the size of the matrix
    n,m = similarity_matrix.shape
    
    for j in range(m):
        # Get the indices of the values greater than the threshold
        valid_indices = np.where(similarity_matrix[:, j] > threshold)[0]
        
        if k is not None and len(valid_indices) > k:
            # Sort valid indices based on the similarity values in descending order
            sorted_indices = valid_indices[np.argsort(similarity_matrix[valid_indices, j])[-k:]]
        else:
            sorted_indices = valid_indices
        
        for i in range(n):
            if i not in sorted_indices:
                filtered_matrix[i, j] = 0
            
    
    return filtered_matrix
    
def create_label_matrix(graph, k):
    # Assuming the 'label' feature is a node feature and stored as a tensor
    labels = graph.ndata['label']
    
    # Get the unique labels
    unique_labels = torch.unique(labels)
    
    # Create a binary matrix
    num_nodes = graph.number_of_nodes()
    num_labels = len(unique_labels)
    
    label_matrix = torch.zeros((num_nodes, num_labels), dtype=torch.float32)
    
    # Fill the binary matrix
    for i in range(num_labels):
        label_matrix[:, i] = (labels == unique_labels[i]).float()
    
    # Randomly select k values per column to stay as 1 and set the rest to 0
    for j in range(num_labels):
        ones_indices = torch.nonzero(label_matrix[:, j]).squeeze().tolist()
        if isinstance(ones_indices, int):
           ones_indices = [ones_indices]  
        print(k)   
        if len(ones_indices) > k:
            keep_indices = random.sample(ones_indices, k)
            set_zero_indices = list(set(ones_indices) - set(keep_indices))
            label_matrix[set_zero_indices, j] = 0
    
    return label_matrix

def create_phon_matrix(graph, label_name, phon_idx):
    # Assuming the 'label' feature is a node feature and stored as a tensor
    labels = graph.ndata['label']
    # Get the label names for each node
    labels_names = label_name[labels]
    # Get the unique labels
    unique_labels = torch.unique(labels)
    
    # Create a binary matrix
    num_nodes = graph.number_of_nodes()
    num_phons = len(phon_idx)
    
    phon_matrix = torch.zeros((num_nodes, num_phons), dtype=torch.float32)
    
    # Fill the binary matrix
    for i in range(num_nodes):
        node_label_name = labels_names[i]
        for j in range(num_phons):
            if phon_idx[j] in node_label_name:
                phon_matrix[i, j] = 1.0
     
    return phon_matrix
        
def softmax_prob(method, graph, num_labels, label_name=None, threshold_probability=None, k=None,idx_phon=None):
   
    num_nodes = graph.number_of_nodes()
    
    #softmax_probabilities = torch.zeros((num_nodes, num_labels), dtype=torch.float32)
    
    if method == 'ML':
       #Load the PyTorch model
       acoustic_model = torch.load('models/cnn.pth')
       acoustic_model.eval()  # Set the model to evaluation mode

# Convert node features to a PyTorch tensor
       node_features = torch.tensor(graph.ndata['feat'].cpu().numpy(), dtype=torch.float32)
       node_features = node_features.view(node_features.shape[0],1 , node_features.shape[1], node_features.shape[2])
# Predict softmax probabilities
       with torch.no_grad():  # Disable gradient calculation
          logits = acoustic_model(node_features)
          softmax_probabilities = F.softmax(logits, dim=1)

# Filter the softmax probabilities
       softmax_probabilities = filter_similarity_matrix(softmax_probabilities.numpy(), threshold=threshold_probability, k=k)
      
      
    elif method == 'fixed' :
      softmax_probabilities = create_label_matrix(graph, k)
     
    elif method == 'mixed':
      #Load the PyTorch model
       acoustic_model = torch.load('models/cnn.pth')
       acoustic_model.eval()  # Set the model to evaluation mode

# Convert node features to a PyTorch tensor
       node_features = torch.tensor(graph.ndata['feat'].cpu().numpy(), dtype=torch.float32)
       node_features = node_features.view(node_features.shape[0],1 , node_features.shape[1], node_features.shape[2])
# Predict softmax probabilities
       with torch.no_grad():  # Disable gradient calculation
          logits = acoustic_model(node_features)
          softmax_probabilities = F.softmax(logits, dim=1)
       softmax_probabilities = filter_similarity_matrix(softmax_probabilities.numpy(), threshold=threshold_probability, k=k)
       fixed_probabilities = create_label_matrix(graph, k)
       softmax_probabilities[fixed_probabilities==1] = 1.0
    
    elif method == 'folle': 
      softmax_probabilities = create_phon_matrix(graph, label_name, idx_phon)
      
    
    
    elif method == 'dnn':
       # Load the PyTorch model (DNN)
      acoustic_model = torch.load('models/dense.pth')
      acoustic_model.eval()  # Set the model to evaluation mode

      # Convert node features to a PyTorch tensor
      node_features = torch.tensor(graph.ndata['feat'].cpu().numpy(), dtype=torch.float32)

# No need to reshape, since node_features are now vectors
# node_features shape: [num_nodes, feature_dim]

# Predict softmax probabilities
      with torch.no_grad():  # Disable gradient calculation
         logits = acoustic_model(node_features)  # Pass the node feature vectors directly into the DNN
         softmax_probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities

# Filter the softmax probabilities (assuming you have a filter_similarity_matrix function)
      softmax_probabilities = filter_similarity_matrix(softmax_probabilities.numpy(), threshold=threshold_probability, k=k)
      
    logging.info(f'{method} method for connection')
    return softmax_probabilities 
      
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--twa', help='threshold for mixte graph', required=True)    
    parser.add_argument('--num_n', help='number of neigheibors for word', required=True)   
    parser.add_argument('--num_n_ac', help='number of neigheibors for acoustic', required=True)
    parser.add_argument('--k_out', help='number of negative neigheibors for acoustic', required=True)
    parser.add_argument('--method', help='', required=False) 
    parser.add_argument('--method_sim', help='', required=False) 
    parser.add_argument('--method_acou', help='', required=False) 
    parser.add_argument('--msw', help='method to compute a word similarity', required=False)
    parser.add_argument('--sub_units', help='fraction of data', required=True) 
    parser.add_argument('--dataset', help='name of dataset', required=True)
    args = parser.parse_args()
    
    save_graph_dir = os.path.join('saved_graphs',args.dataset,args.method_sim, args.method_acou)
# Load the simple DGL graphs
    if not os.path.isfile(os.path.join(save_graph_dir, args.msw, f"kws_graph_{args.num_n_ac}_{args.k_out}_{args.sub_units}.dgl")):
      print('no')
    glist1, label_dict = load_graphs(os.path.join(save_graph_dir,f"kws_graph_{args.num_n_ac}_{args.k_out}_{args.sub_units}.dgl"))
    glist2, _ = load_graphs(os.path.join('saved_graphs',args.dataset, "dgl_words_graph.bin"))

    dgl_G_acoustic = glist1[0]
    dgl_G_words = glist2[0]

    graph1 = dgl_G_acoustic
    graph2 = dgl_G_words

# Extract nodes and edges from graph1
    src1, dst1 = graph1.edges()
    edge_weights1 = graph1.edata['weight']  # Assume edge weights are stored in 'weight'

# Extract nodes and edges from graph2
    src2, dst2 = graph2.edges()
    edge_weights2 = graph2.edata['weight']  # Assume edge weights are stored in 'weight'

# Step 1: Compute the softmax probabilities from a model
# Example softmax probabilities for each acoustic node (random for demonstration purposes)
    num_acoustic_nodes = graph1.num_nodes()

    num_word_nodes = graph2.num_nodes()

    with open(f'label_names_{args.dataset}.pkl', 'rb') as f:
        all_label_names = pickle.load(f)
   
    if  os.path.isfile(f'phon_idx_{args.dataset}.pkl'):
       with open(f'phon_idx_{args.dataset}.pkl', 'rb') as f:
          phon_idx = pickle.load(f)
       idx_phon = {v: k for k, v in phon_idx.items()}
    else :
       idx_phon = None

# Step 2: Define the threshold probability
    threshold_probability = float(args.twa)
    k= math.floor(int(args.num_n))

    softmax_probabilities=softmax_prob(
     method = 'folle' if args.msw == 'phon_coo' else args.method, 
    graph = graph1, num_labels = num_word_nodes, label_name = all_label_names, threshold_probability=threshold_probability,k= k, idx_phon=idx_phon)

    np.save('filtered_softmax_probabilities_words_acoustic.npy', softmax_probabilities)
    
# Step 3: Create links between acoustic and word nodes based on probabilities exceeding the threshold
    links_acoustic_word = []
    probabilities_acoustic_word = []

    

    for i in range(softmax_probabilities.shape[0]):
        for j in range(softmax_probabilities.shape[1]):
            if softmax_probabilities[i, j] > 0:
                 links_acoustic_word.append((i, j))
                 probabilities_acoustic_word.append(softmax_probabilities[i, j])

    print(len(links_acoustic_word))
                
# Combine the edges into a data dictionary for the heterogeneous graph
    data_dict = {
    ('acoustic', 'sim_tic', 'acoustic'): (src1, dst1),
    ('word', 'sim_w', 'word'): (src2, dst2),
    ('acoustic', 'related_to', 'word'): (torch.tensor([src for src, dst in links_acoustic_word]), torch.tensor([dst for src, dst in links_acoustic_word]))
}

    
# Create the heterogeneous graph
    hetero_graph = dgl.heterograph(data_dict)

# Add node features and labels
    hetero_graph.nodes['acoustic'].data['feat'] = graph1.ndata['feat']
    hetero_graph.nodes['word'].data['feat'] = graph2.ndata['feat']
    hetero_graph.nodes['acoustic'].data['label'] = graph1.ndata['label']
#hetero_graph.nodes['word'].data['label'] = graph2.ndata['label']

# Add edge weights
    hetero_graph.edges['sim_tic'].data['weight'] = edge_weights1
    hetero_graph.edges['sim_w'].data['weight'] = edge_weights2
    hetero_graph.edges['related_to'].data['weight'] = torch.tensor(probabilities_acoustic_word)


# Flatten the features of the acoustic nodes
    acoustic_features = hetero_graph.nodes['acoustic'].data['feat']
    flattened_acoustic_features = acoustic_features.view(acoustic_features.shape[0], -1)  # Flatten the features

# Determine the length of the flattened features
    flattened_length = flattened_acoustic_features.shape[1]

# Pad the features of the word nodes to match the flattened length of acoustic features
    word_features = hetero_graph.nodes['word'].data['feat']
    word_feat_shape = word_features.shape
    padded_word_features = torch.zeros((word_feat_shape[0], flattened_length))  # Initialize padded feature tensor

# Copy existing word features into the padded tensor
    padded_word_features[:, :word_feat_shape[1]] = word_features

# Assign the padded features back to the word nodes in the graph
    hetero_graph.nodes['word'].data['feat'] = padded_word_features
    hetero_graph.nodes['acoustic'].data['feat'] = flattened_acoustic_features 
# Print the heterogeneous graph to verify
    print(hetero_graph)


# Define the directory to save the graph
    save_dir = os.path.join('saved_graphs',args.dataset,args.method_sim, args.method_acou,args.method,args.msw)

# Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

# Save the heterogeneous graph
    dgl.save_graphs(os.path.join(save_dir, f"hetero_graph_{args.num_n_ac}_{args.k_out}_{args.num_n}_{args.sub_units}.dgl"), hetero_graph)
