import numpy as np
import networkx as nx
import dgl
import pickle
import torch
import os 
# Load the filtered similarity matrix with labels
matrix_with_labels = np.load('filtered_similarity_matrix_with_labels.npy')

# Extract the labels and similarity matrix
labels = matrix_with_labels[:, 0]
filtered_similarity_matrix = matrix_with_labels[:, 1:]

# Load the saved subset_spectrogram
subset_spectrogram = np.load('subset_spectrogram.npy')


G = nx.Graph()
num_nodes = filtered_similarity_matrix.shape[0]
G.add_nodes_from(range(num_nodes))

# Add edges based on similarity matrix
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        similarity = filtered_similarity_matrix[i, j]
        if similarity > 0:  # Only add edges for non-zero similarity
            
            G.add_edge(i, j, weight=similarity)
            

# Convert NetworkX graph to DGL graph

def build_dgl_graph(nx_graph):
    # Extract edges with weights from the NetworkX graph
    edges = np.array(list(nx_graph.edges(data='weight', default=1.0)), dtype=object)
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    weights = edges[:, 2].astype(float)

    # Create a DGL graph from the edges
    dgl_graph = dgl.graph((src, dst))

    # Add edge weights to the DGL graph
    dgl_graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return dgl_graph
# Copy edge weights from NetworkX graph to DGL graph
dgl_G = build_dgl_graph(G)

dgl_G.ndata['label'] = torch.tensor(labels,dtype=torch.long)
dgl_G.ndata['feat'] =  torch.stack([torch.from_numpy(spec) for spec in subset_spectrogram])

# Example usage: print number of nodes and edges
print("Number of nodes:", dgl_G.number_of_nodes())
print("Number of edges:", dgl_G.number_of_edges())

# Define the directory to save the graph
save_dir = "saved_graphs"
dgl.save_graphs(os.path.join(save_dir,"kws_graph.dgl"), [dgl_G])

