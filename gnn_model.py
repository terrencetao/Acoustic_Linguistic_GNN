import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dgl.data.utils import load_graphs
import argparse 
import os
import pickle
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import networkx as nx
import dgl

class CNN(nn.Module):
    def __init__(self, conv_param, hidden_units):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_param[0][0], out_channels=conv_param[1], kernel_size=conv_param[0][1], padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=conv_param[2])
        self.flatten = nn.Flatten()
        self.input_shape = conv_param[0][2]

        # Détermination de la taille de l'entrée des couches linéaires
        num_conv_features = self._calculate_conv_features(conv_param)
        self.linear_layers = self._create_linear_layers(num_conv_features, hidden_units)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)

        # Appliquer les couches linéaires
        for layer in self.linear_layers:
            x = layer(x)
            x = self.relu(x)
        return x

   
    def _calculate_conv_features(self, conv_param):
        # Calculer le nombre de caractéristiques extraites par les couches de convolution
        dummy_input = torch.zeros((1,conv_param[0][0], *self.input_shape))  # Exemple d'entrée (taille arbitraire)
        conv_output = self.conv1(dummy_input)
        conv_output = self.relu(conv_output)
        conv_output = self.pool(conv_output)
        conv_output = self.flatten(conv_output)
        return conv_output.size(1)
       

    def _create_linear_layers(self, num_conv_features, hidden_units):
        # Créer des couches linéaires en fonction du nombre de caractéristiques extraites par les couches de convolution
        layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                layers.append(nn.Linear(num_conv_features, hidden_units[i]))
            else:
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
        return nn.ModuleList(layers)


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, conv_param,hidden_units):
        super(GCN, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn = CNN(conv_param=conv_param, hidden_units=hidden_units)
        #self.conv1 = SAGEConv(hidden_units[-1], hidden_size, 'mean')
        self.conv1 = SAGEConv(15996, hidden_size, 'mean')
        self.conv2 = SAGEConv(hidden_size, num_classes, 'mean')

    def forward(self, g, features, edge_weights):
        #x = self.cnn(features.unsqueeze(1)).squeeze(1)
        x = self.flatten(features)
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x

# Define the training function
def train(model, g, features, edge_weights, labels, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features, edge_weights)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            


# Define a custom topological loss function
def topological_loss(embeddings, adj_matrix):
    # Calculate pairwise cosine similarity between embeddings
    cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # Zero out the diagonal of the cosine similarity matrix
    cosine_sim = cosine_sim - torch.diag_embed(torch.diag(cosine_sim))
    # Compute the reconstruction loss
    reconstruction_loss = F.mse_loss(cosine_sim, adj_matrix)
    
    return reconstruction_loss

# Define the training function with topological loss
def train_with_topological_loss(model, g, features, edge_weights,adj_matrix, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features, edge_weights)
        loss = topological_loss(embeddings, adj_matrix)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')




parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help ='source folder')
parser.add_argument('--graph_file', help ='graph for trainning')
args = parser.parse_args()
input_folder = args.input_folder    
graph_file = args.graph_file




glist, label_dict = load_graphs(os.path.join(input_folder,graph_file))
dgl_G = glist[0]  

features = dgl_G.ndata['feat']
labels = dgl_G.ndata['label']
edge_weights = dgl_G.edata['weight']

# Initialize the GCN model
in_feats = dgl_G.ndata['feat'].shape[1]
hidden_size = 64
num_classes = len(labels.unique())  # Number of unique labels
conv_param = [
    # Paramètres de la première couche de convolution
    (1, 3, (20,64)),  # Tuple: (nombre de canaux d'entrée, taille du noyau, forme de l'entrée)
    32,
    # Paramètres de la couche de pooling
    (2)
]

print(num_classes)
hidden_units = [32, 32]
model1 = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
model2 = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)



# Train the model

train(model1, dgl_G, features,edge_weights, labels)

# Define the file path for saving the model
model_path = os.path.join('models',"gnn_model.pth")

# Save the model
torch.save(model1.state_dict(), model_path)






# Train the model with topological loss
# Assume adj_matrix is the adjacency matrix of the graph
adj_matrix = torch.tensor(nx.to_numpy_matrix(dgl.to_networkx(dgl_G)))

adj_matrix = adj_matrix.float()
features = features.float()
train_with_topological_loss(model2, dgl_G, features, edge_weights,adj_matrix)
model_path_sup = os.path.join('models',"gnn_model_unsup.pth")
torch.save(model2.state_dict(), model_path_sup)
