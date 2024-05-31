
import numpy as np
import networkx as nx
import dgl
import pickle
import torch
import os

    
def build_dgl_graph(nx_graph):
    # Extract edges with weights from the NetworkX graph
    edges = np.array(list(nx_graph.edges(data='weight', default=1.0)), dtype=object)
    if len(edges) > 0:
        src = edges[:, 0].astype(int)
        dst = edges[:, 1].astype(int)
        weights = edges[:, 2].astype(float)
    else:
        src = np.array([], dtype=int)
        dst = np.array([], dtype=int)
        weights = np.array([], dtype=float)
    
    # Create a DGL graph from the edges
    dgl_graph = dgl.graph((src, dst), num_nodes=nx_graph.number_of_nodes())

    # Add edge weights to the DGL graph
    if len(weights) > 0:
        dgl_graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    else:
        dgl_graph.edata['weight'] = torch.tensor([], dtype=torch.float32)
    
    return dgl_graph
    
# Load the filtered similarity matrix with labels
similarity_matrix = np.load('filtered_similarity_matrix_word.npy')
word_embeddings=np.load('word_embedding.npy')
print(similarity_matrix.shape)
# Load label_names from the file to verify
with open('subset_label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)
label_names = list(label_names)    
    
# Create a NetworkX graph from the similarity matrix with weights
nx_graph = nx.from_numpy_array(similarity_matrix)


# Convert NetworkX graph to a DGL graph without edge attributes
dgl_graph = build_dgl_graph(nx_graph)
# Example usage: print number of nodes and edges
print("Number of nodes:", dgl_graph.number_of_nodes())
print("Number of edges:", dgl_graph.number_of_edges())


# Add node features and labels to DGL graph
dgl_graph.ndata['feat'] = torch.tensor(word_embeddings, dtype=torch.float32)
dgl_graph.ndata['label'] = torch.tensor([label_names.index(label) for label in label_names], dtype=torch.long)

# Save the DGL graph
save_dir = "saved_graphs"

dgl.save_graphs(os.path.join(save_dir,'dgl_words_graph.bin'), [dgl_graph])

print("word graph computed successfully.")
