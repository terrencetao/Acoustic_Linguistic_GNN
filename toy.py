import torch
import random
import Levenshtein

# Simuler un petit graphe
class DummyGraph:
    def __init__(self, labels):
        self.ndata = {'label': torch.tensor(labels)}
    def number_of_nodes(self):
        return len(self.ndata['label'])

# Dictionnaire label -> transcription textuelle
label_to_text = {
    0: "Afa’a Téla",
    1: "Ajʉ’ɛ́ ŋgɔ̄ŋ",
    2: "Akia",
}

# Graph de 6 nœuds
graph = DummyGraph(labels=[0, 0, 1, 1, 2, 2])

def compute_similarity_matrix(label_to_text):
    labels = list(label_to_text.keys())
    n = len(labels)
    S = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                S[i, j] = 1.0
            else:
                dist = Levenshtein.distance(label_to_text[i], label_to_text[j])
                S[i, j] = 1.0 / (1.0 + dist)
    return S

def create_weighted_label_matrix(graph, label_to_text):
    labels = graph.ndata['label']
    unique_labels = torch.unique(labels)
    num_nodes = graph.number_of_nodes()
    num_labels = len(unique_labels)

    label_matrix = torch.zeros((num_nodes, num_labels), dtype=torch.float32)

    # Étape 1 : remplir les vrais labels avec 1
    for i, label in enumerate(unique_labels):
        label_matrix[:, i] = (labels == label).float()

    # Étape 2 : Matrice de similarité
    similarity_matrix = compute_similarity_matrix(label_to_text)

    # Étape 3 : compléter avec des poids de similarité
    weighted_matrix = torch.zeros_like(label_matrix)

    for node_idx in range(num_nodes):
        true_label = labels[node_idx].item()
        for label_idx in range(num_labels):
            if label_matrix[node_idx, label_idx] == 1.0:
                weighted_matrix[node_idx, label_idx] = 1.0
            else:
                weighted_matrix[node_idx, label_idx] = similarity_matrix[true_label, label_idx]

    return weighted_matrix, label_matrix





weighted_label_matrix,  label_matrix = create_weighted_label_matrix(graph, label_to_text)
print(weighted_label_matrix)
print(label_matrix)

