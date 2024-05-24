import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv
import argparse 
import os
import networkx as nx

# Define the HeteroGCN model
class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(HeteroGCN, self).__init__()
        self.conv1 = HeteroGraphConv({
            'sim_tic': GraphConv(in_feats['acoustic'], hidden_size),
            'sim_w': GraphConv(in_feats['word'], hidden_size),
            'related_to': GraphConv(in_feats['acoustic'], hidden_size)
        }, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            'sim_tic': GraphConv(hidden_size, out_feats),
            'sim_w': GraphConv(hidden_size, out_feats),
            'related_to': GraphConv(hidden_size, out_feats)
        }, aggregate='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h

# Define a custom topological loss function
def topological_loss(embeddings_acoustic, embeddings_word, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word):
    # Calculate pairwise cosine similarity between embeddings
    cosine_sim_acoustic = F.cosine_similarity(embeddings_acoustic.unsqueeze(1), embeddings_acoustic.unsqueeze(0), dim=2)
    cosine_sim_word = F.cosine_similarity(embeddings_word.unsqueeze(1), embeddings_word.unsqueeze(0), dim=2)
    
    # Zero out the diagonal of the cosine similarity matrices
    cosine_sim_acoustic = cosine_sim_acoustic - torch.diag_embed(torch.diag(cosine_sim_acoustic))
    cosine_sim_word = cosine_sim_word - torch.diag_embed(torch.diag(cosine_sim_word))
    
    # Compute the reconstruction loss for intra-type connections
    reconstruction_loss_acoustic = F.mse_loss(cosine_sim_acoustic, adj_matrix_acoustic)
    reconstruction_loss_word = F.mse_loss(cosine_sim_word, adj_matrix_word)
    
    # Compute the reconstruction loss for inter-type connections
    reconstruction_loss_acoustic_word = F.mse_loss(
        torch.matmul(embeddings_acoustic, embeddings_word.t()),
        adj_matrix_acoustic_word
    )
    
    return reconstruction_loss_acoustic + reconstruction_loss_word + reconstruction_loss_acoustic_word

# Define the training function with topological loss
def train_with_topological_loss(model, g, features, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        loss = topological_loss(
            embeddings['acoustic'], embeddings['word'], 
            adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Main function to load graph, prepare data, and train model
def main(input_folder, graph_file, epochs):
    # Load the heterogeneous graph
    glist, _ = dgl.load_graphs(os.path.join(input_folder, graph_file))
    hetero_graph = glist[0]
    
    # Extract features
    features = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
        'word': hetero_graph.nodes['word'].data['feat']
    }
    
    # Initialize the HeteroGCN model
    in_feats = {'acoustic': features['acoustic'].shape[1], 'word': features['word'].shape[1]}
    hidden_size = 64
    out_feats = 16  # Set output feature size
    model = HeteroGCN(in_feats, hidden_size, out_feats)
    
    # Extract adjacency matrices
    adj_matrix_acoustic = torch.tensor(nx.to_numpy_matrix(hetero_graph['acoustic', 'sim_tic', 'acoustic'].to_networkx()))
    adj_matrix_word = torch.tensor(nx.to_numpy_matrix(hetero_graph['word', 'sim_w', 'word'].to_networkx()))
    
    # Build the adjacency matrix for acoustic-word relations
    num_acoustic_nodes = hetero_graph.num_nodes('acoustic')
    num_word_nodes = hetero_graph.num_nodes('word')
    adj_matrix_acoustic_word = torch.zeros(num_acoustic_nodes, num_word_nodes)
    
    src, dst = hetero_graph.edges(etype=('acoustic', 'related_to', 'word'))
    adj_matrix_acoustic_word[src, dst] = 1 

    # Convert adjacency matrices to float
    adj_matrix_acoustic = adj_matrix_acoustic.float()
    adj_matrix_word = adj_matrix_word.float()
    adj_matrix_acoustic_word = adj_matrix_acoustic_word.float()

    # Convert features to float
    features = {k: v.float() for k, v in features.items()}
    
 
    # Train the model with topological loss
    train_with_topological_loss(
        model, hetero_graph, features, 
        adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word,
        epochs
    )
    
    # Save the model
    model_path = os.path.join('models', "hetero_gnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

# Parse arguments and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='source folder', required=True)
    parser.add_argument('--graph_file', help='graph for training', required=True)
    parser.add_argument('--epochs', help='number of epochs', required=True)
    args = parser.parse_args()
    
    main(args.input_folder, args.graph_file, int(args.epochs))

