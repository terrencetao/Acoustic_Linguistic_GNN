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


# CNN Model Definition
class MultiLabelCNN(nn.Module):
    def __init__(self, input_shape, num_outputs):  # num_outputs = nb de phonèmes
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        flattened_size = 64 * (input_shape[1] // 4) * (input_shape[2] // 4)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 32, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 64, H/4, W/4)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)  # Sigmoid pour sortie multi-label

    def num_flat_features(self, x):
        return x[0].numel()


def train_multilabel_cnn(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}')


def evaluate_multilabel_cnn(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs', required=True)
    parser.add_argument('--dataset', help='name of dataset', required=True)
    parser.add_argument('--method_sim', help='', required=True)
    parser.add_argument('--sub_units', help='fraction of data', required=True)  

    args = parser.parse_args()
    sub_units = args.sub_units

    matrix_dir = os.path.join('saved_matrix', args.dataset, args.method_sim)

    spectrograms = np.load(os.path.join(matrix_dir, f'subset_spectrogram_{sub_units}.npy'))
    val_spectrograms = np.load(os.path.join(matrix_dir, f'subset_val_spectrogram_{sub_units}.npy'))

    train_word_labels = np.load(os.path.join(matrix_dir, f'subset_label_{sub_units}.npy'))
    val_word_labels = np.load(os.path.join(matrix_dir, f'subset_val_label_{sub_units}.npy'))

    word_embedding = np.load(f'word_embedding_{args.dataset}.npy')  # (Nb de mots, Nb de phonèmes)

    # Convertir spectrogrammes en 4D (N, 1, H, W) si nécessaire
    if spectrograms.ndim == 3:
        spectrograms = np.expand_dims(spectrograms, axis=1)
    if val_spectrograms.ndim == 3:
        val_spectrograms = np.expand_dims(val_spectrograms, axis=1)

    # Vecteurs multi-label de phonèmes pour chaque échantillon
    train_phoneme_vectors = word_embedding[train_word_labels]
    val_phoneme_vectors = word_embedding[val_word_labels]

    X_train = torch.tensor(spectrograms, dtype=torch.float32)
    y_train = torch.tensor(train_phoneme_vectors, dtype=torch.float32)

    X_val = torch.tensor(val_spectrograms, dtype=torch.float32)
    y_val = torch.tensor(val_phoneme_vectors, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_shape = X_train.shape[1:]  # (1, H, W)
    num_outputs = y_train.shape[1]   # Nombre de phonèmes

    model = MultiLabelCNN(input_shape, num_outputs)
    criterion = nn.MSELoss()  # Peut aussi utiliser nn.BCELoss() si binaire
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    train_multilabel_cnn(model, train_loader, criterion, optimizer, num_epochs=int(args.epochs))

    # Évaluation
    val_loss = evaluate_multilabel_cnn(model, val_loader, criterion)
    logging.info(f'Validation Loss (MSE): {val_loss:.4f}')

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    torch.save(model, 'models/multilabel_cnn.pth')

