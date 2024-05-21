
import numpy as np
import networkx as nx
import dgl
import pickle
import torch
import os


def build_dgl_graph(nx_graph):
    # Créer un graphe DGL vide
    dgl_graph = dgl.DGLGraph()

    # Ajouter des nœuds au graphe DGL
    dgl_graph.add_nodes(nx_graph.number_of_nodes())

    # Ajouter des arêtes au graphe DGL avec les poids
    src, dst, weights = zip(*[(int(src), int(dst), data['weight']) for src, dst, data in nx_graph.edges(data=True)])
    dgl_graph.add_edges(src, dst)
    dgl_graph.edata['weight'] = torch.tensor(weights,dtype=torch.float)

    return dgl_graph
    
# Load the filtered similarity matrix with labels
similarity_matrix = np.load('filtered_similarity_matrix_word.npy')
word_embeddings=np.load('word_embedding.npy')

# Load label_names from the file to verify
with open('label_names.pkl', 'rb') as f:
    label_names = pickle.load(f)
label_names = list(label_names)    
    
# Create a NetworkX graph from the similarity matrix with weights
nx_graph = nx.from_numpy_array(similarity_matrix)


# Convert NetworkX graph to a DGL graph without edge attributes
dgl_graph = build_dgl_graph(nx_graph)


# Add node features and labels to DGL graph
dgl_graph.ndata['feat'] = torch.tensor(word_embeddings, dtype=torch.float32)
dgl_graph.ndata['label'] = torch.tensor([label_names.index(label) for label in label_names], dtype=torch.long)

# Save the DGL graph
save_dir = "saved_graphs"

dgl.save_graphs(os.path.join(save_dir,'dgl_words_graph.bin'), [dgl_graph])

print("word graph computed successfully.")
