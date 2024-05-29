import dgl
from dgl.data.utils import load_graphs
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import numpy as np




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
    
parser = argparse.ArgumentParser()
parser.add_argument('--twa', help='threshold for mixte graph', required=True)    
parser.add_argument('--num_n', help='number of neigheibors for word', required=True)   
args = parser.parse_args()
    
save_graph_dir = 'saved_graphs'
# Load the simple DGL graphs
glist1, label_dict = load_graphs(os.path.join(save_graph_dir,"kws_graph.dgl"))
glist2, _ = load_graphs(os.path.join(save_graph_dir, "dgl_words_graph.bin"))

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

acoustic_model = tf.keras.models.load_model('models/model.keras')
softmax_probabilities = acoustic_model.predict(tf.convert_to_tensor(graph1.ndata['feat'].cpu().numpy()))

# Step 2: Define the threshold probability
threshold_probability = float(args.twa)
k= int(args.num_n)

softmax_probabilities=filter_similarity_matrix(softmax_probabilities,  threshold=threshold_probability, k=k)

np.save('filtered_softmax_probabilities_words_acoustic.npy', softmax_probabilities)
# Step 3: Create links between acoustic and word nodes based on probabilities exceeding the threshold
links_acoustic_word = []
probabilities_acoustic_word = []


for i in range(num_acoustic_nodes):
    for j in range(num_word_nodes):
        links_acoustic_word.append((i, j))
        probabilities_acoustic_word.append(softmax_probabilities[i, j])


                
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
hetero_graph.nodes['word'].data['label'] = graph2.ndata['label']

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
save_dir = "saved_graphs"

# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Save the heterogeneous graph
dgl.save_graphs(os.path.join(save_dir, "hetero_graph.dgl"), hetero_graph)
