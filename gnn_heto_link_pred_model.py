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
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IdentityConv(nn.Module):
    def __init__(self, out_dim):
        super(IdentityConv, self).__init__()
        self.out_dim = out_dim

    def forward(self, graph, feat, edge_weight=None):
        if isinstance(feat, tuple):
            # On prend les features de destination (feat_dst)
            return feat[1][:, :self.out_dim]
        else:
            return feat[:, :self.out_dim]

class HeteroLinkGCN(nn.Module):
    def __init__(self, in_feats, hidden_size,nombre_phon, linear_hidden_size):
        super(HeteroLinkGCN, self).__init__()

        self.conv1 = HeteroGraphConv({
            'sim_tic': SAGEConv(in_feats['acoustic'], nombre_phon, 'mean'),
            'sim_w': IdentityConv(out_dim=nombre_phon),
            'related_to': SAGEConv(in_feats['acoustic'], nombre_phon, 'mean')
        }, aggregate='mean')

        self.conv2 = HeteroGraphConv({
            'sim_tic': SAGEConv(hidden_size, nombre_phon, 'mean'),
            'sim_w': IdentityConv(out_dim=nombre_phon),
            'related_to': SAGEConv(hidden_size, nombre_phon, 'mean')
        }, aggregate='mean')

        # MLP for binary classification (link exists or not)
        #self.edge_predictor = nn.Sequential(
        #    nn.Linear(4 * 128, linear_hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(linear_hidden_size, 1),
        #    nn.Sigmoid()  # Sortie entre 0 et 1
        #)

    def forward(self, g, inputs):
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        #h = {k: F.relu(v) for k, v in h.items()}
        #h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        return h  # embeddings

  


def train_link_prediction(model, g, features, true_edge_labels, src, dst, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, epochs=100, lr=0.0002, lamb=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCELoss()  # puisque on applique sigmoid dans le modèle

    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        pred_probs = predict_edge_probabilities_dot(model, embeddings['acoustic'], embeddings['word'], src, dst)

        classification_loss = criterion(pred_probs, true_edge_labels)
        topo_loss = topological_loss(embeddings['acoustic'], embeddings['word'], adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word)
        loss = classification_loss + lamb * topo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step(classification_loss)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, BCE Loss: {classification_loss.item():.4f}")

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
    
    

def generate_negative_edges(g, num_samples, src_type='word', dst_type='acoustic', etype='related_to'):
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

def predict_edge_probabilities(model, acoustic_embeddings, word_embeddings, src, dst):
    src_embed = acoustic_embeddings[src]
    dst_embed = word_embeddings[dst]
    diff = torch.abs(src_embed - dst_embed)
    prod = src_embed * dst_embed
    edge_features = torch.cat([src_embed, dst_embed, diff, prod], dim=1)

    predicted_probs = model.edge_predictor(edge_features).squeeze()
    return predicted_probs

def predict_edge_probabilities_dot(model, acoustic_embeddings, word_embeddings, src, dst):
    dst_embed = acoustic_embeddings[dst]
    src_embed = word_embeddings[src]

    score = (src_embed * dst_embed).sum(dim=1)  # Dot product
    prob = torch.sigmoid(score)
    return prob
    
def main(input_folder, graph_file, epochs, lamb, dataset):
    glist, _ = dgl.load_graphs(os.path.join(input_folder, graph_file))
    hetero_graph = glist[0]

    features = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
        'word': hetero_graph.nodes['word'].data['feat']
    }
    with open(f'phon_idx_{dataset}.pkl', 'rb') as f:
        phon_idx = pickle.load(f)
    
    # Get edge data
    # Positifs
    src_pos, dst_pos = hetero_graph.edges(etype=('word', 'related_to', 'acoustic'))
    weights_pos = torch.ones(len(src_pos))

    # Négatifs (autant que les positifs, ou plus/moins si tu veux)
    src_neg, dst_neg = generate_negative_edges(hetero_graph, num_samples=len(src_pos))
    weights_neg = torch.zeros(len(src_neg))

    # Fusionne
    src = torch.cat([src_pos, src_neg], dim=0)
    dst = torch.cat([dst_pos, dst_neg], dim=0)
    true_edge_weights = torch.cat([weights_pos, weights_neg], dim=0)
    perm = torch.randperm(len(src))
    src, dst, true_edge_labels = src[perm], dst[perm], true_edge_weights[perm]  # shuffle

    # Graph meta info
    in_feats = {ntype: features[ntype].shape[1] for ntype in features}
    hidden_size = 512
    linear_hidden_size = 64
    nb_phon = len(phon_idx)
    model = HeteroLinkGCN(in_feats, hidden_size, nb_phon, linear_hidden_size)

    # Build adjacency matrices for topological loss
    adj_matrix_acoustic = torch.tensor(nx.to_numpy_array(hetero_graph['acoustic', 'sim_tic', 'acoustic'].to_networkx()))
    adj_matrix_word = torch.tensor(nx.to_numpy_array(hetero_graph['word', 'sim_w', 'word'].to_networkx()))
    num_acoustic = hetero_graph.num_nodes('acoustic')
    num_word = hetero_graph.num_nodes('word')
    adj_matrix_acoustic_word = torch.zeros(num_word, num_acoustic)
    adj_matrix_acoustic_word[src, dst] = 1

    adj_matrix_acoustic = adj_matrix_acoustic.float()
    adj_matrix_word = adj_matrix_word.float()
    adj_matrix_acoustic_word = adj_matrix_acoustic_word.float()
    features = {k: v.float() for k, v in features.items()}
    true_edge_weights = true_edge_weights.float()

    model = train_link_prediction(
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
    parser.add_argument('--dataset', help='name of dataset', required=True)
    args = parser.parse_args()

    main(args.input_folder, args.graph_file, int(args.epochs), float(args.lamb), args.dataset)




