import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, SAGEConv
import argparse 
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the HeteroGCN model
class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(HeteroGCN, self).__init__()
        self.conv1 = HeteroGraphConv({
            'sim_tic': SAGEConv(in_feats['acoustic'], hidden_size, 'mean'),
            'sim_w': SAGEConv(in_feats['word'], hidden_size, 'mean'),
            'related_to': SAGEConv(in_feats['acoustic'], hidden_size, 'mean')
        }, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            'sim_tic': SAGEConv(hidden_size, out_feats, 'mean'),
            'sim_w': SAGEConv(hidden_size, out_feats, 'mean'),
            'related_to': SAGEConv(hidden_size, out_feats, 'mean')
        }, aggregate='mean')

    def forward(self, g, inputs):
        # Get edge weights
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        return h

# Contrastive loss function
def contrastive_loss(pos_score, neg_score):
    loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean() - torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
    return loss

# Training with unsupervised loss
def train_with_unsupervised_loss(model, g, features, negative_samples=5, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        # Get the node embeddings from the model
        embeddings = model(g, features)
        
        # For each edge type, compute the positive and negative scores
        pos_scores = {}
        neg_scores = {}
        
        for etype in g.etypes:
            src, dst = g.edges(etype=etype)
            pos_scores[etype] = torch.sum(embeddings[g.to_canonical_etype(etype)[0]][src] * embeddings[g.to_canonical_etype(etype)[2]][dst], dim=1)
            
            # Negative sampling: randomly select negative pairs
            neg_src = src
            neg_dst = torch.randint(0, g.num_nodes(g.to_canonical_etype(etype)[2]), (len(src),))
            neg_scores[etype] = torch.sum(embeddings[g.to_canonical_etype(etype)[0]][neg_src] * embeddings[g.to_canonical_etype(etype)[2]][neg_dst], dim=1)
        
        # Compute the contrastive loss
        total_loss = 0
        for etype in pos_scores:
            loss = contrastive_loss(pos_scores[etype], neg_scores[etype])
            total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item()}')

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
    
    # Convert features to float
    features = {k: v.float() for k, v in features.items()}
    
    # Train the model with unsupervised loss
    train_with_unsupervised_loss(
        model, hetero_graph, features,
        negative_samples=5, epochs=epochs, lr=0.01
    )
    
    # Save the model
    model_path = os.path.join('models', "hetero_gnn_model_unsupervised.pth")
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

