import os
import torch
import dgl
import numpy as np
import csv
from gnn_heto_model import HeteroGCN
from gnn_model import GCN
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse


def load_graphs(path):
    return dgl.load_graphs(path)
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--twa', help='word similarity threshold', required=True)
parser.add_argument('--num_n_h', help='method to compute a word similarity', required=True)
parser.add_argument('--mhg', help='method to compute a word similarity', required=True)
parser.add_argument('--num_n_a', help='method to compute a word similarity', required=True)
parser.add_argument('--ta', help='method to compute a word similarity', required=True)
parser.add_argument('--alpha', help='method to compute a word similarity', required=True)
parser.add_argument('--tw', help='method to compute a word similarity', required=True)
parser.add_argument('--msw', help='method to compute a word similarity', required=True)
parser.add_argument('--msa', help='method to compute heterogeneous similarity', required=True)
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
parser.add_argument('--sub_units', help='fraction of data', required=True)  
args = parser.parse_args()


# Paths
graph_folder = 'saved_graphs'
model_folder = 'models'

# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder, "kws_graph.dgl"))
dgl_G = glist[0]
features = dgl_G.ndata['feat']
labels = dgl_G.ndata['label']

# Define the input features size
in_feats = features[0].shape[0] * features[0].shape[1]
hidden_size = 64
num_classes = len(torch.unique(labels))
conv_param = [(1, 3, (20, 64)), 32, 2]
hidden_units = [32, 32]

# Load supervised GCN model
model_sup_path = os.path.join(model_folder, "gnn_model.pth")
loaded_model_sup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_sup.load_state_dict(torch.load(model_sup_path))

edge_weights = dgl_G.edata['weight']

with torch.no_grad():
    loaded_model_sup.eval()
    node_embeddings_sup = loaded_model_sup(dgl_G, features, edge_weights).numpy()

# Load unsupervised GCN model
model_unsup_path = os.path.join(model_folder, "gnn_model_unsup.pth")
loaded_model_unsup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_unsup.load_state_dict(torch.load(model_unsup_path))

# Load the heterogeneous graph
glists, _ = dgl.load_graphs(os.path.join(graph_folder, 'hetero_graph.dgl'))
hetero_graph = glists[0]

# Load the heterogeneous GCN model
features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
in_feats = {'acoustic': features_dic['acoustic'].shape[1], 'word': features_dic['word'].shape[1]}
hidden_size = 64
out_feats = 16

# Initialize the model
model = HeteroGCN(in_feats, hidden_size, out_feats)

# Load the pre-trained model state
model.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model.pth")))
model.eval()

# Extract acoustic node representations
with torch.no_grad():
    embeddings = model(hetero_graph, features_dic)
    acoustic_embeddings = embeddings['acoustic']

# Extract labels for training
labels_np = labels.numpy()

# Function to split data into train and test sets
def train_test_split_data(embeddings, labels, test_size=0.2, random_state=42):
    return train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)

# Function to train and evaluate SVM and return accuracy
def train_evaluate_svm(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split_data(embeddings, labels)
    
    # Initialize the SVM model
    clf = svm.SVC(kernel='linear')
    # Train the model
    clf.fit(X_train, y_train)
    # Predict on test data
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# CSV file path
csv_file = f'accuracy_results_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Supervised Model', 'Unsupervised Model', 'Heterogeneous Model', 'Spectrogram Baseline', 'twa'  ,'num_n_h' ,'mhg'  ,'num_n_a' ,'ta' ,'alpha' ,'tw' ,'msw','msa'])

# Embeddings from supervised model
node_embeddings_sup = torch.from_numpy(node_embeddings_sup)
node_embeddings_sup = node_embeddings_sup.numpy()

# Train and evaluate SVM for supervised embeddings
accuracy_sup = train_evaluate_svm(node_embeddings_sup, labels_np)

# Train and evaluate SVM for unsupervised embeddings
node_embeddings_unsup = loaded_model_unsup(dgl_G, features, edge_weights).detach().numpy()
accuracy_unsup = train_evaluate_svm(node_embeddings_unsup, labels_np)

# Train and evaluate SVM for heterogeneous model embeddings
acoustic_embeddings_np = acoustic_embeddings.detach().numpy()
accuracy_hetero = train_evaluate_svm(acoustic_embeddings_np, labels_np)

# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms

spectrograms = np.load('subset_spectrogram.npy')
flattened_spectrograms = flatten_spectrograms(spectrograms)

# Train and evaluate SVM for spectrogram embeddings
accuracy_spectrogram = train_evaluate_svm(flattened_spectrograms, labels_np)

# Write accuracy results to CSV file
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy_sup, accuracy_unsup, accuracy_hetero, accuracy_spectrogram, float(args.twa)  ,float(args.num_n_h) ,args.mhg  , float(args.num_n_a) ,float(args.ta) ,float(args.alpha) ,float(args.tw) ,args.msw, args.msa])

