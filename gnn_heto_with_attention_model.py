import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GATConv
import argparse 
import os
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the HeteroGCN model with attention for all relations
class HeteroGCNWithAllAttention(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_heads):
        super(HeteroGCNWithAllAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.conv1 = HeteroGraphConv({
            'sim_tic': GATConv(in_feats['acoustic'], hidden_size, num_heads, feat_drop=0.6, attn_drop=0.6, activation=F.elu),
            'sim_w': GATConv(in_feats['word'], hidden_size, num_heads, feat_drop=0.6, attn_drop=0.6, activation=F.elu),
            'related_to': GATConv(in_feats['acoustic'], hidden_size, num_heads, feat_drop=0.6, attn_drop=0.6, activation=F.elu)
        }, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            'sim_tic': GATConv(hidden_size * num_heads, out_feats, 1, feat_drop=0.6, attn_drop=0.6, activation=None),
            'sim_w': GATConv(hidden_size * num_heads, out_feats, 1, feat_drop=0.6, attn_drop=0.6, activation=None),
            'related_to': GATConv(hidden_size * num_heads, out_feats, 1, feat_drop=0.6, attn_drop=0.6, activation=None)
        }, aggregate='mean')

    def forward(self, g, inputs):
        # Get edge weights
        edge_weights = {etype: g.edges[etype].data['weight'].float() for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        h = {k: v.view(v.size(0), -1) for k, v in h.items()}  # Flatten the output for all relations
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        h = {k: v.view(v.size(0), -1) for k, v in h.items()}  # Flatten the output for all relations
        return h


    
def contrastive_loss(embeddings, adj_matrix, margin=1.0):
    # Calculate pairwise cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # Zero out the diagonal of the cosine similarity matrix
    cosine_sim = cosine_sim - torch.diag_embed(torch.diag(cosine_sim))
    
    # Generate pairs of indices for positive and negative examples
    pos_pairs = torch.nonzero(adj_matrix == 1, as_tuple=False)
    neg_pairs = torch.nonzero(adj_matrix == 0, as_tuple=False)
    
    # Compute contrastive loss for positive and negative pairs
    loss_pos = torch.mean((1 - cosine_sim[pos_pairs])**2)
    loss_neg = torch.mean(torch.clamp(cosine_sim[neg_pairs] - margin, min=0)**2)
    
    # Total loss
    loss = loss_pos + loss_neg
    return loss
    
def train_with_contrastive_loss(model, g, features, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        loss = contrastive_loss(
            embeddings['acoustic'], adj_matrix_acoustic,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


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
    num_head = 4
    model = HeteroGCNWithAllAttention(in_feats, hidden_size, out_feats, num_head)
    
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
    
    # Train the model with contrastive loss
    train_with_contrastive_loss(
        model, hetero_graph, features, 
        adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word,
        epochs
    )
    
    # Save the model
    model_path = os.path.join('models', "hetero_gcn_with_attention_model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')

# Parse arguments and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='source folder', required=True)
    parser.add_argument('--graph_file', help='graph for training', required=True)
    parser.add_argument('--epochs', help='number of epochs', required=True)
    args = parser.parse_args()
    
    main(args.input_folder, args.graph_file, int(args.epochs))


# Parse arguments and run main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='source folder', required=True)
    parser.add_argument('--graph_file', help='graph for training', required=True)
    parser.add_argument('--epochs', help='number of epochs', required=True, type=int)
    args = parser.parse_args()
    
    main(args.input_folder, args.graph_file, args.epochs)

