import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, SAGEConv
import argparse
import os
import networkx as nx
import logging
from torch.optim import lr_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HeteroLinkGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, linear_hidden_size):
        super(HeteroLinkGCN, self).__init__()

        self.conv1 = HeteroGraphConv({
            'sim_tic': SAGEConv(in_feats['acoustic'], hidden_size, 'mean'),
            'sim_w': SAGEConv(in_feats['word'], hidden_size, 'mean'),
            'related_to': SAGEConv(in_feats['acoustic'], hidden_size, 'mean')
        }, aggregate='mean')

        self.conv2 = HeteroGraphConv({
            'sim_tic': SAGEConv(hidden_size, 128, 'mean'),
            'sim_w': SAGEConv(hidden_size, 128, 'mean'),
            'related_to': SAGEConv(hidden_size, 128, 'mean')
        }, aggregate='mean')

        # MLP to predict edge weight
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * 128, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, 1)
        )

    def forward(self, g, inputs):
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        return h  # embeddings

def generate_negative_edges(g, num_samples, src_type='acoustic', dst_type='word', etype='related_to'):
    src_nodes = torch.arange(g.num_nodes(src_type))
    dst_nodes = torch.arange(g.num_nodes(dst_type))
    existing_edges = set(zip(*g.edges(etype=etype)))
    
    negatives = []
    while len(negatives) < num_samples:
        src = torch.randint(0, g.num_nodes(src_type), (1,)).item()
        dst = torch.randint(0, g.num_nodes(dst_type), (1,)).item()
        if (src, dst) not in existing_edges:
            negatives.append((src, dst))
            existing_edges.add((src, dst))  # pour éviter les doublons
    return torch.tensor([i[0] for i in negatives]), torch.tensor([i[1] for i in negatives])

def predict_edge_weights(model, acoustic_embeddings, word_embeddings, src, dst):
    src_embed = acoustic_embeddings[src]
    dst_embed = word_embeddings[dst]
    edge_features = torch.cat([src_embed, dst_embed], dim=1)
    predicted_weights = model.edge_predictor(edge_features).squeeze()
    return predicted_weights


def train_edge_regression(model, g, features, true_edge_weights, src, dst, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, epochs=100, lr=0.001, lamb=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        pred_weights = predict_edge_weights(model, embeddings['acoustic'], embeddings['word'], src, dst)

        regression_loss = F.mse_loss(pred_weights, true_edge_weights)
        topo_loss = topological_loss(embeddings['acoustic'], embeddings['word'], adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word)
        loss = regression_loss + lamb * topo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, MSE: {regression_loss.item():.4f}")

    return model


def topological_loss(embeddings_acoustic, embeddings_word,
                     adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, lamb_1=0):
    # Acoustic similarity
    cosine_sim_acoustic = F.cosine_similarity(
        embeddings_acoustic.unsqueeze(1), embeddings_acoustic.unsqueeze(0), dim=2)
    cosine_sim_acoustic = cosine_sim_acoustic - torch.diag_embed(torch.diag(cosine_sim_acoustic))
    loss_acoustic = F.mse_loss(cosine_sim_acoustic, adj_matrix_acoustic)

    # Word similarity
    cosine_sim_word = F.cosine_similarity(
        embeddings_word.unsqueeze(1), embeddings_word.unsqueeze(0), dim=2)
    cosine_sim_word = cosine_sim_word - torch.diag_embed(torch.diag(cosine_sim_word))
    loss_word = F.mse_loss(cosine_sim_word, adj_matrix_word)

   

    return loss_acoustic + lamb_1*loss_word



def main(input_folder, graph_file, epochs, lamb):
    glist, _ = dgl.load_graphs(os.path.join(input_folder, graph_file))
    hetero_graph = glist[0]

    features = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
        'word': hetero_graph.nodes['word'].data['feat']
    }

    # Get edge data
    # Positifs
    src_pos, dst_pos = hetero_graph.edges(etype=('acoustic', 'related_to', 'word'))
    weights_pos = hetero_graph.edges['related_to'].data['weight'].squeeze()

    # Négatifs (autant que les positifs, ou plus/moins si tu veux)
    src_neg, dst_neg = generate_negative_edges(hetero_graph, num_samples=len(src_pos))
    weights_neg = torch.zeros(len(src_neg))

    # Fusionne
    src = torch.cat([src_pos, src_neg], dim=0)
    dst = torch.cat([dst_pos, dst_neg], dim=0)
    true_edge_weights = torch.cat([weights_pos, weights_neg], dim=0)

    # Graph meta info
    in_feats = {ntype: features[ntype].shape[1] for ntype in features}
    hidden_size = 512
    linear_hidden_size = 64

    model = HeteroLinkGCN(in_feats, hidden_size, linear_hidden_size)

    # Build adjacency matrices for topological loss
    adj_matrix_acoustic = torch.tensor(nx.to_numpy_array(hetero_graph['acoustic', 'sim_tic', 'acoustic'].to_networkx()))
    adj_matrix_word = torch.tensor(nx.to_numpy_array(hetero_graph['word', 'sim_w', 'word'].to_networkx()))
    num_acoustic = hetero_graph.num_nodes('acoustic')
    num_word = hetero_graph.num_nodes('word')
    adj_matrix_acoustic_word = torch.zeros(num_acoustic, num_word)
    adj_matrix_acoustic_word[src, dst] = 1

    adj_matrix_acoustic = adj_matrix_acoustic.float()
    adj_matrix_word = adj_matrix_word.float()
    adj_matrix_acoustic_word = adj_matrix_acoustic_word.float()
    features = {k: v.float() for k, v in features.items()}
    true_edge_weights = true_edge_weights.float()

    model = train_edge_regression(
        model, hetero_graph, features, true_edge_weights, src, dst,
        adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word,
        epochs, lamb=lamb
    )

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', "hetero_gnn_edge_regressor.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='source folder', required=True)
    parser.add_argument('--graph_file', help='graph for training', required=True)
    parser.add_argument('--lamb', help='hyperparameter for objective function', required=True)
    parser.add_argument('--epochs', help='number of epochs', required=True)
    args = parser.parse_args()

    main(args.input_folder, args.graph_file, int(args.epochs), float(args.lamb))

