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
parser.add_argument('--mgw', help='method to build word graph ', required=True)
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


num_heads = 4
model_attention_path = os.path.join(model_folder, "hetero_gnn_with_all_attention_model.pth")
model_attention = HeteroGCNWithAllAttention(in_feats, hidden_size, out_feats, num_heads=num_heads)
model_attention.load_state_dict(torch.load(model_attention_path))
model_attention.eval()

# Extract acoustic node representations
with torch.no_grad():
    embeddings_attention = model_attention(hetero_graph, features_dic)
    acoustic_embeddings_attention = embeddings_attention['acoustic']


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

os.makedirs('accuracy', exist_ok=True)
# CSV file path
csv_file = f'accuracy_results_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Supervised Model', 'Unsupervised Model', 'Heterogeneous Model','Heterogeneous attention Model', 'Spectrogram Baseline', 'CNN Model', 'twa', 'num_n_h', 'mhg', 'num_n_a', 'ta', 'alpha', 'tw', 'msw', 'msa', 'mgw'])

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

# Train and evaluate SVM on the new heterogeneous attention model embeddings
accuracy_attention = train_evaluate_svm(acoustic_embeddings_attention.numpy(), labels_np)
print(f"Accuracy of the Heterogeneous Attention Model: {accuracy_attention:.4f}")

# Spectrogram baseline embeddings
def flatten_spectrograms(spectrograms):
    num_samples = spectrograms.shape[0]
    flattened_spectrograms = spectrograms.reshape(num_samples, -1)
    return flattened_spectrograms

spectrograms = np.load('subset_spectrogram.npy')
flattened_spectrograms = flatten_spectrograms(spectrograms)

# Train and evaluate SVM for spectrogram embeddings
accuracy_spectrogram = train_evaluate_svm(flattened_spectrograms, labels_np)

# Prepare data for CNN
spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
labels_tensor = torch.tensor(labels_np, dtype=torch.long)
spectrograms_tensor = spectrograms_tensor.unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(spectrograms_tensor, labels_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# Define and train the CNN model
input_shape = spectrograms_tensor.shape[1:]  # (1, height, width)
cnn_model = SimpleCNN(input_shape, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
train_cnn(cnn_model, train_loader, criterion, optimizer)
accuracy_cnn = evaluate_cnn(cnn_model, test_loader)
print(f'CNN Model Accuracy: {accuracy_cnn}')

# Write accuracy results to CSV file
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([accuracy_sup, accuracy_unsup, accuracy_hetero, accuracy_attention,accuracy_spectrogram, accuracy_cnn, float(args.twa), float(args.num_n_h), args.mhg, float(args.num_n_a), float(args.ta), float(args.alpha), float(args.tw), args.msw, args.msa, args.mgw])

