import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dense Model Definition
class SimpleDense(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleDense, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_shape), 128)  # Flatten the input shape for Dense layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Softmax for multi-class classification

def train_dense(model, train_loader, criterion, optimizer, num_epochs=20):
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

def evaluate_dense(model, test_loader):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs', required=True)
    parser.add_argument('--dataset', help='name of dataset', required=False)
    parser.add_argument('--method_sim', help='', required=True)
    parser.add_argument('--sub_units', help='fraction of data', required=True)

    args = parser.parse_args()
    sub_units = args.sub_units

    matrix_dir = os.path.join('saved_matrix', args.dataset, args.method_sim)
    labels_np = np.load(os.path.join(matrix_dir, f'subset_label_{sub_units}.npy'))
    val_labels_np = np.load(os.path.join(matrix_dir, f'subset_val_label_{sub_units}.npy'))
    subset_val_spectrograms = np.load(os.path.join(matrix_dir, f'subset_val_spectrogram_{sub_units}.npy'))
    spectrograms = np.load(os.path.join(matrix_dir, f'subset_spectrogram_{sub_units}.npy'))
    num_classes = len(np.unique(labels_np))

    # Prepare data for Dense model
    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
    val_spectrograms_tensor = torch.tensor(subset_val_spectrograms, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_np, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels_np, dtype=torch.long)
    spectrograms_tensor = spectrograms_tensor.unsqueeze(1)  # Keep it as 1 channel
    val_spectrograms_tensor = val_spectrograms_tensor.unsqueeze(1)

    X_train, X_test, y_train, y_test = spectrograms_tensor, val_spectrograms_tensor, labels_tensor, val_labels_tensor
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Define and train the Dense model
    logging.info(f'train the Dense model')
    input_shape = spectrograms_tensor.shape[1:]  # (1, height, width)
    dense_model = SimpleDense(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dense_model.parameters(), lr=0.001)
    train_dense(dense_model, train_loader, criterion, optimizer, num_epochs=int(args.epochs))
    accuracy_dense = evaluate_dense(dense_model, test_loader)
    logging.info(f'Dense Model Accuracy: {accuracy_dense}')

    torch.save(dense_model, 'models/dense.pth')

