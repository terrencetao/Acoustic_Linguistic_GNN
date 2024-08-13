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
from generate_similarity_matrix_acoustic import compute_distance_for_pair, compute_dtw_distance
from heterogenous_graph import filter_similarity_matrix
import logging
from joblib import Parallel, delayed
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)
    

def add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, distance_function, n_jobs=-1):
    num_existing_nodes = dgl_G.number_of_nodes()
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new nodes to the graph
    dgl_G.add_nodes(num_new_nodes)

    # Add features for the new nodes
    new_features = torch.from_numpy(new_node_spectrograms)
    dgl_G.ndata['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = new_features

    # Randomly select k nodes from the existing graph for each new node
    existing_indices = np.arange(num_existing_nodes)
    
    def process_new_node(new_node_index):
        selected_indices = np.random.choice(existing_indices, k, replace=False)
        new_node_spectrogram = new_node_spectrograms[new_node_index - num_existing_nodes]
        
        edges = []
        for i in selected_indices:
            distance = distance_function(new_node_spectrogram, dgl_G.ndata['feat'][i].numpy())
            similarity = np.exp(-distance)
            edges.append((new_node_index, i, similarity))
            edges.append((i, new_node_index, similarity))
        return edges

    # Use joblib to parallelize the processing of new nodes
    all_edges = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in range(num_existing_nodes, num_existing_nodes + num_new_nodes))
    
    # Flatten the list of edges and add them to the graph
    for edges in all_edges:
        for src, dst, weight in edges:
            dgl_G.add_edges(src, dst, {'weight': torch.tensor([weight], dtype=torch.float32)})
    
    return dgl_G, num_existing_nodes
    
    
def add_new_acoustic_nodes_to_hetero_graph(hetero_graph, new_node_spectrograms, k, distance_function, ml_model, threshold_probability, n_jobs=-1):
    num_existing_acoustic_nodes = hetero_graph.num_nodes('acoustic')
    num_existing_word_nodes = hetero_graph.num_nodes('word')
    num_new_nodes = new_node_spectrograms.shape[0]
    
    # Add new 'acoustic' nodes to the graph
    hetero_graph.add_nodes(num_new_nodes, ntype='acoustic')
    
    # Add features for the new 'acoustic' nodes
    flattened_spectrograms = flatten_spectrograms(new_node_spectrograms)
    new_features = torch.from_numpy(flattened_spectrograms)
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = new_features

    # Get softmax probabilities for connections to 'word' nodes using the ML model
    ml_predictions = ml_model.predict(new_node_spectrograms)
    ml_probabilities = torch.from_numpy(ml_predictions)
    
    # Filter probabilities based on the threshold
    ml_probabilities = filter_similarity_matrix(ml_probabilities.numpy(), threshold=threshold_probability, k=k)
    
    def process_new_node(new_node_index):
        edges = []
        probabilities = []
        for word_node_index in range(num_existing_word_nodes):
            similarity = ml_probabilities[new_node_index - num_existing_acoustic_nodes, word_node_index]
            if similarity > 0:
                edges.append((new_node_index, word_node_index))
                probabilities.append(similarity)
        return edges, probabilities

    # Use joblib to parallelize the processing of new nodes
    all_edges_and_probs = Parallel(n_jobs=n_jobs)(delayed(process_new_node)(new_node_index) for new_node_index in range(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes))
    
    # Flatten the list of edges and add them to the graph
    all_edges = [edge for edges, _ in all_edges_and_probs for edge in edges]
    all_probabilities = [prob for _, probs in all_edges_and_probs for prob in probs]
    
    if all_edges:
        src, dst = zip(*all_edges)
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        hetero_graph.add_edges(src, dst, etype=('acoustic', 'related_to', 'word'))
        new_weights = torch.tensor(all_probabilities, dtype=torch.float32)
        if 'weight' not in hetero_graph.edges['related_to'].data:
           num_existing_edges = hetero_graph.num_edges(('acoustic', 'related_to', 'word'))
           hetero_graph.edges['related_to'].data['weight'] = torch.zeros(num_existing_edges, dtype=torch.float32)
    
    # Assign new edge weights to the new edges only
        hetero_graph.edges['related_to'].data['weight'][-new_weights.shape[0]:] = new_weights

        #hetero_graph.edges['related_to'].data['weight'][num_existing_acoustic_nodes:num_existing_acoustic_nodes  + num_new_nodes] = torch.tensor(all_probabilities, dtype=torch.float32)
    return hetero_graph, num_existing_acoustic_nodes
    
    
    
def generate_embeddings(gcn_model, dgl_G, new_node_spectrograms, k, compute_distance):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    dgl_G (dgl.DGLGraph): The graph structure containing existing nodes.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    
    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new nodes to the graph and get the updated graph and the number of existing nodes
    dgl_G, num_existing_nodes = add_new_nodes_to_graph_randomly(dgl_G, new_node_spectrograms, k, compute_distance)
    
    # Extract edge weights and features from the graph
    edge_weights = dgl_G.edata['weight']
    features = dgl_G.ndata['feat']
    
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        embeddings = gcn_model(dgl_G, features, edge_weights).numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_nodes, num_existing_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    # Debugging prints
    print(embeddings.shape)
    print(new_node_indices)
    
    return embeddings[num_existing_nodes:], new_node_embeddings


def generate_embeddings_hetero(gcn_model, hetero_graph, new_node_spectrograms, k, compute_distance, ml_model, threshold_probability, n_jobs=-1):
    """
    Generate embeddings for new nodes using the provided GCN model.
    
    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    hetero_graph (dgl.DGLHeteroGraph): The heterogeneous graph structure.
    new_node_spectrograms (np.ndarray): Spectrograms of the new nodes to be added.
    k (int): The number of neighbors to connect each new node to.
    compute_distance (function): A function to compute the distance between nodes.
    ml_model (tf.keras.Model): The ML model for predicting connection probabilities.
    threshold_probability (float): The threshold for filtering connection probabilities.
    n_jobs (int): The number of jobs for parallel processing.

    Returns:
    np.ndarray: Embeddings for the new nodes.
    """
    # Add new 'acoustic' nodes to the graph
    hetero_graph, num_existing_acoustic_nodes = add_new_acoustic_nodes_to_hetero_graph(
        hetero_graph, 
        new_node_spectrograms, 
        k, 
        compute_distance, 
        ml_model, 
        threshold_probability, 
        n_jobs
    )
    
    # Extract features from the graph
    features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
    # Generate embeddings using the GCN model
    with torch.no_grad():
        gcn_model.eval()
        # Assuming the GCN model takes the graph and node features as input
        embeddings = gcn_model(hetero_graph, features_dic)
        embeddings = embeddings['acoustic'].numpy()
    
    # Extract the embeddings for the new nodes
    num_new_nodes = new_node_spectrograms.shape[0]
    new_node_indices = np.arange(num_existing_acoustic_nodes, num_existing_acoustic_nodes + num_new_nodes)
    new_node_embeddings = embeddings[new_node_indices]
    
    return embeddings[num_existing_acoustic_nodes:], new_node_embeddings
    
# Function to split data into train and test sets
def train_test_split_data(embeddings, labels, test_size=0.2, random_state=42):
    return train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)

# Function to train and evaluate SVM and return accuracy
def train_evaluate_svm( X_train, X_test, y_train, y_test):
    #X_train, X_test, y_train, y_test = train_test_split_data(embeddings, labels)
    
    # Initialize the SVM model
    clf = svm.SVC(kernel='linear')
    # Train the model
    clf.fit(X_train, y_train)
    # Predict on test data
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_cnn(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

def evaluate_cnn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total  

# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms
        
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
parser.add_argument('--drop_freq', help='dim frequency ', required=False)  
parser.add_argument('--drop_int', help='dim amplitude ', required=False) 
parser.add_argument('--sub_units', help='fraction of data', required=True)  
parser.add_argument('--dataset', help='name of dataset', required=True)
args = parser.parse_args()


# Paths
graph_folder = 'saved_graphs'
model_folder = 'models'

# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder, "kws_graph.dgl"))
dgl_G = glist[0]

features = dgl_G.ndata['feat']
labels = dgl_G.ndata['label']
subset_val_labels = np.load('subset_val_label.npy')
subset_val_spectrograms = np.load('subset_val_spectrogram.npy')

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

logging.info(f'Extract acoustic node representations')
node_embeddings_sup, node_val_embeddings_sup = generate_embeddings(gcn_model=loaded_model_sup, 
                                                dgl_G=dgl_G, new_node_spectrograms=subset_val_spectrograms, 
                                                k=int(args.num_n_a),compute_distance=compute_dtw_distance)

print(node_embeddings_sup.shape)
print(node_val_embeddings_sup.shape)

# Load unsupervised GCN model
logging.info(f'Load unsupervised GCN model')
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
logging.info(f'Extract acoustic node representations')
acoustic_model = tf.keras.models.load_model('models/model.keras')
acoustic_embeddings, acoustic_val_embeddings = generate_embeddings_hetero(gcn_model=model, 
                                                hetero_graph=hetero_graph, new_node_spectrograms=subset_val_spectrograms, 
                                                k=int(args.num_n_a),compute_distance=compute_dtw_distance, ml_model=acoustic_model, 
                                                threshold_probability=float(args.twa), n_jobs=-1)



# Extract labels for training
labels_np = labels.numpy()
val_labels_np = subset_val_labels

num_heads = 4
logging.info(f'Load unsupervised GCN attention model')
#model_attention_path = os.path.join(model_folder, "hetero_gcn_with_attention_model.pth")
#model_attention = HeteroGCNWithAllAttention(in_feats, hidden_size, out_feats, num_heads=num_heads)
#model_attention.load_state_dict(torch.load(model_attention_path))
#model_attention.eval()

# Extract acoustic node representations
logging.info(f'Extract acoustic node representations')

    
#acoustic_embeddings_attention, acoustic_val_embeddings_attention = generate_embeddings_hetero(gcn_model=model_attention, 
#                                                hetero_graph=hetero_graph, new_node_spectrograms=subset_val_spectrograms, 
#                                                k=int(args.num_n_a),compute_distance=compute_dtw_distance, ml_model=acoustic_model, 
#                                                threshold_probability=float(args.twa), n_jobs=-1)
   


os.makedirs('accuracy', exist_ok=True)
#dataset
dataset = args.dataset
os.makedirs(f'accuracy/{dataset}', exist_ok=True)

# CSV file path
csv_file = f'{dataset}/accuracy_val_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Supervised Model', 'Unsupervised Model', 'Heterogeneous Model','Heterogeneous attention Model', 'Spectrogram Baseline', 'CNN Model', 'twa', 'num_n_h', 'mhg', 'num_n_a', 'ta', 'alpha', 'tw', 'msw', 'msa', 'mgw'])



# Train and evaluate SVM for supervised embeddings
logging.info(f'Train and evaluate SVM for supervised embeddings')
accuracy_sup = train_evaluate_svm( X_train=node_embeddings_sup, X_test=node_val_embeddings_sup, y_train=labels_np, y_test=val_labels_np)
 
# Train and evaluate SVM for unsupervised embeddings
logging.info(f'Train and evaluate SVM for unsupervised embeddings')
node_embeddings_unsup, node_val_embeddings_unsup = generate_embeddings(gcn_model=loaded_model_unsup, 
                                                dgl_G=dgl_G, new_node_spectrograms=subset_val_spectrograms, 
                                                k=int(args.num_n_a),compute_distance=compute_dtw_distance)

accuracy_unsup = train_evaluate_svm(X_train=node_embeddings_unsup, X_test=node_val_embeddings_unsup, y_train=labels_np, y_test=val_labels_np)

# Train and evaluate SVM for heterogeneous model embeddings
logging.info(f'Train and evaluate SVM for heterogeneous model embeddings')

accuracy_hetero = train_evaluate_svm(X_train=acoustic_embeddings, X_test=acoustic_val_embeddings, y_train=labels_np, y_test=val_labels_np)
logging.info(f"Accuracy of the Heterogeneous Model: {accuracy_hetero:.4f}")

# Train and evaluate SVM on the new heterogeneous attention model embeddings
logging.info(f'Train and evaluate SVM on the new heterogeneous attention model embeddings')
#accuracy_attention = train_evaluate_svm(acoustic_embeddings_attention.numpy(), labels_np)
accuracy_attention =0.0
logging.info(f"Accuracy of the Heterogeneous Attention Model: {accuracy_attention:.4f}")



logging.info(f'Train and evaluate SVM for spectrogram embeddings')

spectrograms = np.load('subset_spectrogram.npy')
flattened_spectrograms = flatten_spectrograms(spectrograms)
flattened_val_spectrograms = flatten_spectrograms(subset_val_spectrograms)
# Train and evaluate SVM for spectrogram embeddings
accuracy_spectrogram = train_evaluate_svm( X_train=flattened_spectrograms, X_test=flattened_val_spectrograms, y_train=labels_np, y_test=val_labels_np)

# Prepare data for CNN
spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
val_spectrograms_tensor = torch.tensor(subset_val_spectrograms, dtype=torch.float32)
labels_tensor = torch.tensor(labels_np, dtype=torch.long)
val_labels_tensor = torch.tensor(val_labels_np, dtype=torch.long)
spectrograms_tensor = spectrograms_tensor.unsqueeze(1) 
val_spectrograms_tensor = val_spectrograms_tensor.unsqueeze(1)
#X_train, X_test, y_train, y_test = train_test_split(spectrograms_tensor, labels_tensor, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = spectrograms_tensor, val_spectrograms_tensor, labels_tensor, val_labels_tensor
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define and train the CNN model
logging.info(f'train the CNN model')
input_shape = spectrograms_tensor.shape[1:]  # (1, height, width)
cnn_model = SimpleCNN(input_shape, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
train_cnn(cnn_model, train_loader, criterion, optimizer)
accuracy_cnn = evaluate_cnn(cnn_model, test_loader)
logging.info(f'CNN Model Accuracy: {accuracy_cnn}')

# Write accuracy results to CSV file
logging.info(f'Write accuracy results to CSV file')
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy_sup, accuracy_unsup, accuracy_hetero, accuracy_attention,accuracy_spectrogram, accuracy_cnn, float(args.twa), float(args.num_n_h), args.mhg, float(args.num_n_a), float(args.ta), float(args.alpha), float(args.tw), args.msw, args.msa, args.mgw])

