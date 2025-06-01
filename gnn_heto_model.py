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

class HeteroGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, linear_hidden_size):
        super(HeteroGCN, self).__init__()

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

        self.linear1 = nn.Linear(hidden_size, linear_hidden_size)
        self.linear2 = nn.Linear(linear_hidden_size, 32)
        self.linear3 = nn.Linear(32, out_feats)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, inputs):
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        #h = {k: F.relu(v) for k, v in h.items()}
        #h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        embeddings = h

        h['acoustic'] = F.relu(self.linear1(h['acoustic']))
        h['acoustic'] = F.relu(self.linear2(h['acoustic']))
        h['acoustic'] = self.linear3(h['acoustic'])

        return h['acoustic'], embeddings

def topological_loss(embeddings_acoustic, embeddings_word, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word):
    cosine_sim_acoustic = F.cosine_similarity(embeddings_acoustic.unsqueeze(1), embeddings_acoustic.unsqueeze(0), dim=2)
    cosine_sim_acoustic = cosine_sim_acoustic - torch.diag_embed(torch.diag(cosine_sim_acoustic))
    reconstruction_loss_acoustic = F.mse_loss(cosine_sim_acoustic, adj_matrix_acoustic)
    return reconstruction_loss_acoustic

def train_with_topological_loss_cross_loss(model, g, features, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, labels, epochs=100, lr=0.001, lamb=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(epochs):
        model.train()
        logits, embeddings = model(g, features)
        loss =  F.cross_entropy(logits, labels) + lamb *topological_loss(
            embeddings['acoustic'], embeddings['word'],
            adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0 or epoch == epochs:
            _, predicted = torch.max(logits, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy*100:.2f}%')

    return model

def main(input_folder, graph_file, epochs, lamb):
    glist, _ = dgl.load_graphs(os.path.join(input_folder, graph_file))
    hetero_graph = glist[0]

    features = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
        'word': hetero_graph.nodes['word'].data['feat']
    }
    labels = hetero_graph.nodes['acoustic'].data['label']
    in_feats = {'acoustic': features['acoustic'].shape[1], 'word': features['word'].shape[1]}
    hidden_size = 512
    linear_hidden_size = 64
    out_feats = len(labels.unique())
    model = HeteroGCN(in_feats, hidden_size, out_feats, linear_hidden_size)

    adj_matrix_acoustic = torch.tensor(nx.to_numpy_array(hetero_graph['acoustic', 'sim_tic', 'acoustic'].to_networkx()))
    adj_matrix_word = torch.tensor(nx.to_numpy_array(hetero_graph['word', 'sim_w', 'word'].to_networkx()))
    num_acoustic_nodes = hetero_graph.num_nodes('acoustic')
    num_word_nodes = hetero_graph.num_nodes('word')
    adj_matrix_acoustic_word = torch.zeros(num_word_nodes, num_acoustic_nodes)
    src, dst = hetero_graph.edges(etype=('word', 'related_to', 'acoustic'))
    adj_matrix_acoustic_word[src, dst] = 1

    adj_matrix_acoustic = adj_matrix_acoustic.float()
    adj_matrix_word = adj_matrix_word.float()
    adj_matrix_acoustic_word = adj_matrix_acoustic_word.float()
    features = {k: v.float() for k, v in features.items()}

    model = train_with_topological_loss_cross_loss(
        model, hetero_graph, features,
        adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, labels,
        epochs, lamb=lamb
    )

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', "hetero_gnn_model.pth")
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

