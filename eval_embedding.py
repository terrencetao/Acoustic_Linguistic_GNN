import os
import torch
import dgl
import numpy as np
import csv
from gnn_heto_model import HeteroGCN
from gnn_heto_with_attention_model import HeteroGCNWithAllAttention
from gnn_model import GCN
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
from weak_ML2 import SimpleCNN, evaluate_cnn, train_cnn
from weakDense import SimpleDense, evaluate_dense, train_dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)
    
    
logging.info(f' ----------------------------------------------------- Evaluation of Representation on training set   -----------------------------------------------')
       
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
parser.add_argument('--mgw', help='method to build word graph ', required=True)
parser.add_argument('--mma', help='method to build acoustic matrix', required=True)
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
parser.add_argument('--sub_units', help='fraction of data', required=True)  
parser.add_argument('--dataset', help='name of dataset', required=True)
args = parser.parse_args()


# Paths
graph_folder = os.path.join('saved_graphs',args.dataset,args.mma,args.msa)
model_folder = 'models'
matrix_folder = os.path.join('saved_matrix',args.dataset, args.mma)
# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder,f"kws_graph_{args.num_n_a}_{args.sub_units}.dgl"))
dgl_G = glist[0]

features = dgl_G.ndata['feat']
print(features.shape)
labels = dgl_G.ndata['label']
subset_val_labels = np.load(os.path.join(matrix_folder,f'subset_val_label_{args.sub_units}.npy'))
subset_val_spectrograms = np.load(os.path.join(matrix_folder,f'subset_val_spectrogram_{args.sub_units}.npy'))

# Define the input features size
in_feats = features[0].shape[0] * features[0].shape[1]

hidden_size = 64
num_classes = len(torch.unique(labels))
conv_param = [(1, 3, (20, 64)), 32, 2]
hidden_units = [32, 32]

# Load supervised GCN model
logging.info(f'Load supervised GCN model')
model_sup_path = os.path.join(model_folder, "gnn_model.pth")
loaded_model_sup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_sup.load_state_dict(torch.load(model_sup_path))

edge_weights = dgl_G.edata['weight']
logging.info(f'Extract acoustic node representations')
with torch.no_grad():
    loaded_model_sup.eval()
    _,node_embeddings_sup = loaded_model_sup(dgl_G, features, edge_weights)
    node_embeddings_sup  = node_embeddings_sup.numpy()
# Load unsupervised GCN model
logging.info(f'Load unsupervised GCN model')
model_unsup_path = os.path.join(model_folder, "gnn_model_unsup.pth")
loaded_model_unsup = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
#loaded_model_unsup.load_state_dict(torch.load(model_unsup_path))

# Load unsupervised sage GCN model
#logging.info(f'Load hibrid GCN model')
model_hibrid_path = os.path.join(model_folder, "gnn_model_hibrid.pth")
loaded_model_hibrid = GCN(in_feats, hidden_size, num_classes, conv_param, hidden_units)
loaded_model_hibrid.load_state_dict(torch.load(model_hibrid_path))

# Load the heterogeneous graph
glists, _ = dgl.load_graphs(os.path.join(graph_folder, args.mhg, args.msw,f"hetero_graph_{args.num_n_a}_{args.num_n_h}_{args.sub_units}.dgl"))
hetero_graph = glists[0]

# Load the heterogeneous GCN model
features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
in_feats = {'acoustic': features_dic['acoustic'].shape[1], 'word': features_dic['word'].shape[1]}
hidden_size = 512
linear_hidden_size = 64
out_feats = len(labels.unique())

# Initialize the model
model = HeteroGCN(in_feats, hidden_size, out_feats, linear_hidden_size)
#model_sage = HeteroGCN(in_feats, hidden_size, out_feats)
# Load the pre-trained model state
model.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model.pth")))
model.eval()

#model_sage.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model_unsupervised.pth")))
#model_sage.eval()
# Extract acoustic node representations
logging.info(f'Extract acoustic node representations')
with torch.no_grad():
    _,embeddings = model(hetero_graph, features_dic)
    acoustic_embeddings = embeddings['acoustic']

#with torch.no_grad():
#    embeddings_sage = model_sage(hetero_graph, features_dic)
#    acoustic_embeddings_sage = embeddings_sage['acoustic']
# Extract labels for training
labels_np = labels.numpy()


num_heads = 4
logging.info(f'Load unsupervised GCN attention model')
#model_attention_path = os.path.join(model_folder, "hetero_gcn_with_attention_model.pth")
#model_attention = HeteroGCNWithAllAttention(in_feats, hidden_size, out_feats, num_heads=num_heads)
#model_attention.load_state_dict(torch.load(model_attention_path))
#model_attention.eval()

# Extract acoustic node representations
logging.info(f'Extract acoustic node representations')
#with torch.no_grad():
#    embeddings_attention = model_attention(hetero_graph, features_dic)
#    acoustic_embeddings_attention = embeddings_attention['acoustic']


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





os.makedirs('accuracy', exist_ok=True)
#dataset
dataset = args.dataset
os.makedirs(f'accuracy/{dataset}', exist_ok=True)

# CSV file path
csv_file = f'{dataset}/accuracy_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Supervised Model', 'Unsupervised Model','Hibrid Model', 'Heterogeneous Model', 'Heterogeneous sage Model','Heterogeneous attention Model', 'Spectrogram Baseline', 'CNN Model', 'DNN Model','twa', 'num_n_h', 'mhg', 'num_n_a', 'ta', 'alpha', 'tw', 'msw', 'msa', 'mgw','mma'])

# Embeddings from supervised model
node_embeddings_sup = torch.from_numpy(node_embeddings_sup)
node_embeddings_sup = node_embeddings_sup.numpy()

# Train and evaluate SVM for supervised embeddings
logging.info(f'Train and evaluate SVM for supervised embeddings')
accuracy_sup = train_evaluate_svm(node_embeddings_sup, labels_np)

# Train and evaluate SVM for unsupervised embeddings
logging.info(f'Train and evaluate SVM for unsupervised embeddings')
_,node_embeddings_unsup = loaded_model_unsup(dgl_G, features, edge_weights)
node_embeddings_unsup = node_embeddings_unsup.detach().numpy()
accuracy_unsup = train_evaluate_svm(node_embeddings_unsup, labels_np)

# Train and evaluate SVM for hibrid embeddings
logging.info(f'Train and evaluate SVM for hibrid embeddings')
_,node_embeddings_hibrid = loaded_model_hibrid(dgl_G, features, edge_weights)
node_embeddings_hibrid = node_embeddings_hibrid.detach().numpy()
accuracy_hibrid = train_evaluate_svm(node_embeddings_hibrid, labels_np)

# Train and evaluate SVM for heterogeneous model embeddings
logging.info(f'Train and evaluate SVM for heterogeneous model embeddings')
acoustic_embeddings_np = acoustic_embeddings.detach().numpy()
accuracy_hetero = train_evaluate_svm(acoustic_embeddings_np, labels_np)
logging.info(f"Accuracy of the Heterogeneous Model: {accuracy_hetero:.4f}")

# Train and evaluate SVM for heterogeneous model embeddings
#logging.info(f'Train and evaluate SVM for heterogeneous sage model embeddings')
#acoustic_embeddings_np_sage = acoustic_embeddings_sage.detach().numpy()
#accuracy_hetero_sage = train_evaluate_svm(acoustic_embeddings_np_sage, labels_np)
accuracy_hetero_sage =0.0
logging.info(f"Accuracy of the Heterogeneous sage Model: {accuracy_hetero_sage:.4f}")

# Train and evaluate SVM on the new heterogeneous attention model embeddings
logging.info(f'Train and evaluate SVM on the new heterogeneous attention model embeddings')
#accuracy_attention = train_evaluate_svm(acoustic_embeddings_attention.numpy(), labels_np)
accuracy_attention =0.0
logging.info(f"Accuracy of the Heterogeneous Attention Model: {accuracy_attention:.4f}")

# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms

spectrograms = np.load(os.path.join(matrix_folder ,f'subset_spectrogram_{args.sub_units}.npy'))
#flattened_spectrograms = flatten_spectrograms(spectrograms)

# Train and evaluate SVM for spectrogram embeddings
#accuracy_spectrogram = train_evaluate_svm(flattened_spectrograms, labels_np)
accuracy_spectrogram = 0.0
# Prepare data for CNN
spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
labels_tensor = torch.tensor(labels_np, dtype=torch.long)
spectrograms_tensor = spectrograms_tensor.unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(spectrograms_tensor, labels_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define and train the CNN model
logging.info(f'train the CNN model')

cnn_model = torch.load('models/cnn.pth')
cnn_model = torch.load('models/cnn.pth')
dnn_model = torch.load('models/dense.pth')
accuracy_cnn = evaluate_cnn(cnn_model, test_loader)
accuracy_dnn = evaluate_dense(dnn_model, test_loader)
logging.info(f'CNN Model Accuracy: {accuracy_cnn}')
logging.info(f'DNN Model Accuracy: {accuracy_dnn}')

# Write accuracy results to CSV file
logging.info(f'Write accuracy results to CSV file')
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy_sup, accuracy_unsup, accuracy_hibrid, accuracy_hetero,accuracy_hetero_sage, accuracy_attention,accuracy_spectrogram, accuracy_cnn, accuracy_dnn, float(args.twa), float(args.num_n_h), args.mhg, float(args.num_n_a), float(args.ta), float(args.alpha), float(args.tw), args.msw, args.msa, args.mgw,args.mma])

