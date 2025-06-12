import os
import torch
import dgl
import numpy as np
import csv
from gnn_heto_model import HeteroGCN
from gnn_heto_link_pred_model import HeteroLinkGCN
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
from generate_similarity_matrix_acoustic import compute_distance_for_pair, distance_dtw, vgg_distance
from heterogenous_graph import filter_similarity_matrix
import logging
from joblib import Parallel, delayed
import tensorflow as tf
from weak_ML2 import SimpleCNN,evaluate_cnn
from tqdm import tqdm
import math
from weakDense import SimpleDense, evaluate_dense, train_dense
import torch.nn.functional as F
import numpy as np
import copy
import pickle
from sklearn.neural_network import MLPClassifier
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graphs(path):
    return dgl.load_graphs(path)
    


    

    
def add_edges_without_duplicates(graph, src, dst, weights, etype):
    existing_edges = set(zip(graph.edges(etype=etype)[0].numpy(), graph.edges(etype=etype)[1].numpy()))
    
    new_edges = []
    new_weights = []
    for s, d, w in zip(src, dst, weights):
        if (s, d) not in existing_edges:
            new_edges.append((s, d))
            new_weights.append(w)

    if new_edges:
        new_src, new_dst = zip(*new_edges)
        graph.add_edges(new_src, new_dst, {'weight': torch.tensor(new_weights, dtype=torch.float32)}, etype=etype)
    


    
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

def add_new_nodes_to_hetero_graph_knn(hetero_graph, new_node_spectrograms, n_jobs=-1):
    """
    Ajoute de nouveaux nœuds 'acoustic' à un graphe hétérogène DGL, en créant des arêtes
    pondérées par similarité cosinus avec les nœuds existants.
    
    Args:
        hetero_graph (dgl.DGLHeteroGraph): Le graphe hétérogène d'entrée.
        new_node_spectrograms (np.ndarray): Nouvelles features (ex: spectrogrammes) [N, ...].
        n_jobs (int): Nombre de processus pour le parallélisme (joblib).
    
    Returns:
        dgl.DGLHeteroGraph: Graphe enrichi.
        int: Nombre de nœuds existants avant ajout.
    """
    num_existing_nodes = hetero_graph.num_nodes('acoustic')
    num_new_nodes = new_node_spectrograms.shape[0]

    # Ajout des nouveaux nœuds
    hetero_graph.add_nodes(num_new_nodes, ntype='acoustic')

    # Conversion des nouvelles features en tenseur et aplatissement
    acoustic_features = torch.from_numpy(new_node_spectrograms)
    flattened_new_features = acoustic_features.view(acoustic_features.shape[0], -1)

    # Ajout des features au graphe
    hetero_graph.nodes['acoustic'].data['feat'][num_existing_nodes:num_existing_nodes + num_new_nodes] = flattened_new_features

    # Extraction des features des anciens nœuds pour calcul de similarité
    existing_features = hetero_graph.nodes['acoustic'].data['feat'][:num_existing_nodes]
    existing_features = existing_features.detach().cpu().numpy()

    # Fonction pour traiter un seul nœud
    def process_new_node(new_node_index):
        rel_idx = new_node_index - num_existing_nodes
        new_embedding = flattened_new_features[rel_idx].detach().cpu().numpy().reshape(1, -1)
        similarities = cosine_similarity(new_embedding, existing_features)[0]

        # Création d'arêtes bidirectionnelles
        edges = [(new_node_index, i, similarities[i]) for i in range(num_existing_nodes)]
        edges += [(i, new_node_index, similarities[i]) for i in range(num_existing_nodes)]
        return edges

    # Parallélisation avec joblib
    all_edges = Parallel(n_jobs=n_jobs)(
        delayed(process_new_node)(idx) for idx in tqdm(range(num_existing_nodes, num_existing_nodes + num_new_nodes))
    )

    # Ajout des arêtes au graphe
    for edges in all_edges:
        src, dst, weights = zip(*edges)
        hetero_graph.add_edges(
            src, dst,
            data={'weight': torch.tensor(weights, dtype=torch.float32)},
            etype=('acoustic', 'sim_tic', 'acoustic')
        )

    return hetero_graph, num_existing_nodes


    



    
  
  
  
    




      
    




    

def generate_embeddings_hetero_regressor(gcn_model, hetero_graph, num_existing_acoustic_nodes):
    """
    Generate embeddings for all acoustic nodes in a heterogeneous graph using the trained GCN model.

    Parameters:
    gcn_model (torch.nn.Module): The trained GCN model.
    hetero_graph (dgl.DGLHeteroGraph): The heterogeneous graph structure.
    num_existing_acoustic_nodes (int): Number of existing acoustic nodes before adding new ones.

    Returns:
    np.ndarray: Tuple of (existing_node_embeddings, new_node_embeddings)
    """
    # Extract node features
    features_dic = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
        'word': hetero_graph.nodes['word'].data['feat']
    }

    with torch.no_grad():
        gcn_model.eval()
        embeddings = gcn_model(hetero_graph, features_dic)
        embeddings_acoustic = embeddings['acoustic'].cpu().numpy()

    # Split existing and new acoustic node embeddings
    existing_embeddings = embeddings_acoustic[:num_existing_acoustic_nodes]
    new_embeddings = embeddings_acoustic[num_existing_acoustic_nodes:]

    return existing_embeddings, new_embeddings
    
    
def evaluate_acoustic_word_link_prediction_topk(gcn_model, hetero_graph,
                                                num_existing_acoustic_nodes,
                                                labels_acoustic, labels_word,encoder,
                                                top_k=5):
    """
    Predict links and compute top-1 and top-K accuracy between new acoustic and word nodes.

    Parameters:
    - gcn_model (HeteroLinkGCN): Trained GNN model.
    - hetero_graph (DGLHeteroGraph): The heterogeneous graph.
    - num_existing_acoustic_nodes (int): Count of original acoustic nodes.
    - labels_acoustic (List[int] or np.ndarray): Labels of new acoustic nodes.
    - labels_word (List[int] or np.ndarray): Labels of word nodes.
    - top_k (int): Top-K value for top-K accuracy.

    Returns:
    - accuracy_top1 (float): Top-1 accuracy.
    - accuracy_topk (float): Top-K accuracy.
    """
    gcn_model.eval()
    device = next(gcn_model.parameters()).device

    with torch.no_grad():
        # Step 1: Compute embeddings
        features_dic = {
            'acoustic': hetero_graph.nodes['acoustic'].data['feat'].to(device),
            'word': hetero_graph.nodes['word'].data['feat'].to(device)
        }
        embeddings = gcn_model(hetero_graph, features_dic)
        emb_acoustic = embeddings['acoustic']
        emb_word = embeddings['word']

        # Step 2: New acoustic embeddings
        new_emb_acoustic = emb_acoustic[num_existing_acoustic_nodes:]
        num_new = new_emb_acoustic.shape[0]
        num_words = emb_word.shape[0]

        # Step 3: Predict scores
        pred_scores = []
        for i in range(num_new):
            acoustic_vec = new_emb_acoustic[i].repeat(num_words, 1)
            diff = torch.abs(acoustic_vec - emb_word)
            prod = acoustic_vec * emb_word
            combined = torch.cat([acoustic_vec, emb_word, diff, prod], dim=1)
            scores = gcn_model.edge_predictor(combined).squeeze()
            pred_scores.append(scores)

        pred_scores = torch.stack(pred_scores)  # [num_new, num_words]

        # Step 4: Top-K predictions
        topk_indices_pred = torch.topk(pred_scores, k=top_k, dim=1).indices.cpu().numpy() # la prediction se fait sur les class reenocoder au moment de la generation de la matrice d'adjacence acoustique
        topk_indices = [encoder.inverse_transform(indices_pred) for indices_pred in topk_indices_pred] # l'on revient a l'encodage originel celui contenu dans le graph word, pour pouvoir comparer 
        correct_top1 = 0
        correct_topk = 0

        true_labels = encoder.inverse_transform(labels_acoustic)
        for i in range(num_new):
            true_label = true_labels[i]
            predicted_word_labels =  topk_indices[i]

            if predicted_word_labels[0] == true_label:
                correct_top1 += 1
            if true_label in predicted_word_labels:
                correct_topk += 1


        accuracy_top1 = correct_top1 / len(labels_acoustic)
        accuracy_topk = correct_topk / len(labels_acoustic)

        return accuracy_top1, accuracy_topk




def evaluate_link_prediction_classification_topk(
    gcn_model,
    hetero_graph,
    num_existing_acoustic_nodes,
    labels_acoustic,
    labels_word,
    encoder,
    top_k=5
):
    """
    Évaluation du modèle pour la prédiction binaire de liens (existe ou pas) entre
    les nouveaux nœuds acoustiques et les nœuds word.

    Retourne top-1 et top-k accuracy.

    Paramètres :
    - gcn_model : le modèle GNN entraîné.
    - hetero_graph : graphe hétérogène DGL.
    - num_existing_acoustic_nodes : nombre de nœuds acoustiques avant ajout.
    - labels_acoustic : labels (classes) des nouveaux nœuds acoustiques.
    - labels_word : labels (classes) des nœuds word.
    - top_k : valeur de K pour l'accuracy top-K.

    Retour :
    - accuracy_top1, accuracy_topk
    """

    gcn_model.eval()
    device = next(gcn_model.parameters()).device

    with torch.no_grad():
        # Étape 1 : Récupération des features
        features_dic = {
            'acoustic': hetero_graph.nodes['acoustic'].data['feat'].to(device),
            'word': hetero_graph.nodes['word'].data['feat'].to(device)
        }

        # Étape 2 : Calcul des embeddings
        embeddings = gcn_model(hetero_graph, features_dic)
        emb_acoustic = embeddings['acoustic']
        emb_word = embeddings['word']

        # Étape 3 : Embeddings des nouveaux nœuds acoustiques
        new_emb_acoustic = emb_acoustic[num_existing_acoustic_nodes:]
        num_new = new_emb_acoustic.shape[0]
        num_words = emb_word.shape[0]

       
        new_emb_acoustic = emb_acoustic[num_existing_acoustic_nodes:]
        num_new = new_emb_acoustic.shape[0]

        # Étape 4 : Prédiction des scores de lien (vectorisée)
        # Produit scalaire : (num_new, dim) @ (dim, num_words) => (num_new, num_words)
        scores = torch.matmul(new_emb_acoustic, emb_word.T)
        pred_scores = torch.sigmoid(scores)  # (num_new, num_words)

        # Étape 5 : top-K prédiction
        # Step 4: Top-K predictions
        topk_indices_pred = torch.topk(pred_scores, k=top_k, dim=1).indices.cpu().numpy() # la prediction se fait sur les class reenocoder au moment de la generation de la matrice d'adjacence acoustique
        topk_indices = [encoder.inverse_transform(indices_pred) for indices_pred in topk_indices_pred] # l'on revient a l'encodage originel celui contenu dans le graph word, pour pouvoir comparer 
       
        correct_top1 = 0
        correct_topk = 0
        true_labels = encoder.inverse_transform(labels_acoustic)
        for i in range(num_new):
            true_label = true_labels[i]
            predicted_word_labels =  topk_indices[i]

            if predicted_word_labels[0] == true_label:
                correct_top1 += 1
            if true_label in predicted_word_labels:
                correct_topk += 1

        accuracy_top1 = correct_top1 / num_new
        accuracy_topk = correct_topk / num_new

        return accuracy_top1, accuracy_topk



    
# Function to split data into train and test sets
def train_test_split_data(embeddings, labels, test_size=0.2, random_state=42):
    return train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)
    

def train_evaluate_mlp(X_train, X_test, y_train, y_test):
    
    # Standardize embeddings
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    

    # MLP with two hidden layers: 128 and 64 neurons
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=300, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    return accuracy_score(y_test, y_pred)
    

def train_evaluate_dnn_pytorch(X_train, X_test, y_train, y_test , num_epochs=50, batch_size=32, lr=1e-3):
    
    

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(labels))

    model = SimpleDense(input_shape=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dense(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluation
    accuracy = evaluate_dense(model, test_loader)
    return accuracy



    
logging.info(f' ----------------------------------------------------- Evaluation of Representation on validate set  -----------------------------------------------')
       
parser = argparse.ArgumentParser()
parser.add_argument('--twa', help='word similarity threshold', required=True)
parser.add_argument('--num_n_h', help='method to compute a word similarity', required=True)
parser.add_argument('--mhg', help='method to compute a word similarity', required=True)
parser.add_argument('--num_n_a', help='method to compute a word similarity', required=True)
parser.add_argument('--k_out', help='method to compute a word similarity', required=True)
parser.add_argument('--k_inf', help='method to compute a word similarity', required=True)
parser.add_argument('--lamb', help ='hyperparameter for objective fonction')
parser.add_argument('--density', help ='desenty intra cluster')
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
parser.add_argument('--add', help='model to add a new node in graph dnn or ML', required=True)
args = parser.parse_args()


# Paths
graph_folder = os.path.join('saved_graphs',args.dataset,args.mma,args.msa)
model_folder = 'models'
matrix_folder = os.path.join('saved_matrix',args.dataset, args.mma)
# Load the homogeneous graph
glist, label_dict = load_graphs(os.path.join(graph_folder,f"kws_graph_{args.num_n_a}_{args.k_out}_{args.sub_units}.dgl"))
dgl_G = glist[0]

features = dgl_G.ndata['feat']

labels = dgl_G.ndata['label']
subset_val_labels = np.load(os.path.join(matrix_folder,f'subset_val_label_{args.sub_units}.npy'))
subset_val_spectrograms = np.load(os.path.join(matrix_folder,f'subset_val_spectrogram_{args.sub_units}.npy'))

# Define the input features size
in_feats = features[0].shape[0] 

hidden_size = 64
num_classes = len(torch.unique(labels))


# Extract labels for training
labels_np = labels.numpy()
val_labels_np = subset_val_labels



# Load the heterogeneous graph
glists, _ = dgl.load_graphs(os.path.join(graph_folder, args.mhg,args.msw,  f"hetero_graph_{args.num_n_a}_{args.k_out}_{args.num_n_h}_{args.sub_units}.dgl"))
hetero_graph = glists[0]

# Load the heterogeneous GCN model
features_dic = {
    'acoustic': hetero_graph.nodes['acoustic'].data['feat'],
    'word': hetero_graph.nodes['word'].data['feat']
}
with open(f'phon_idx_{args.dataset}.pkl', 'rb') as f:
        phon_idx = pickle.load(f)
  
in_feats = {'acoustic': features_dic['acoustic'].shape[1], 'word': features_dic['word'].shape[1]}
hidden_size = hetero_graph.nodes['word'].data['feat'].shape[1]
linear_hidden_size = 64
nb_phon = hetero_graph.nodes['word'].data['feat'].shape[1]
out_feats = len(labels.unique())

# Initialize the model
model = HeteroGCN(in_feats, hidden_size, out_feats, linear_hidden_size)
model_hetero_regressor = HeteroLinkGCN(in_feats, hidden_size,nb_phon, linear_hidden_size)

# Load the pre-trained model state
model.load_state_dict(torch.load(os.path.join(model_folder, "hetero_gnn_model.pth")))
model.eval()


# Extract acoustic node representations
logging.info(f'Extract acoustic node representations from hetero GCN')

hetero_graph_path_val = os.path.join(graph_folder, args.mhg,args.msw, args.add, f"hetero_graph_val_{args.mhg}_{args.num_n_a}_{args.k_out}{args.num_n_h}_{args.sub_units}.dgl")
hetero_regressor_graph_path_val = os.path.join(graph_folder, args.mhg,args.msw, args.add, f"hetero_regressor_graph_val_{args.mhg}_{args.num_n_a}_{args.k_out}{args.num_n_h}_{args.sub_units}.dgl")


                                                
# ADD AND GENERATE EMBEDDING FROM THE REGRESSOR HETERO MODEL                        
                        
if not os.path.isfile(hetero_regressor_graph_path_val):
# Add new 'acoustic' nodes to the graph
  
    hetero_regressor_graph, num_existing_acoustic_nodes = add_new_nodes_to_hetero_graph_knn(
                                copy.deepcopy(hetero_graph), 
                                 new_node_spectrograms=subset_val_spectrograms, 
                                  n_jobs=-1)
    
    
    dgl.save_graphs(hetero_regressor_graph_path_val, hetero_regressor_graph)
else:
    print(f"File {hetero_regressor_graph_path_val} already exists. Skipping computation.")
    num_existing_acoustic_nodes =  hetero_graph.num_nodes('acoustic')
    glist, label_dict = load_graphs(hetero_regressor_graph_path_val)
    hetero_regressor_graph = glist[0]
acoustic_embeddings_regressor, acoustic_val_embeddings_regressor = generate_embeddings_hetero_regressor(gcn_model=model_hetero_regressor, 
                                                hetero_graph=hetero_regressor_graph,num_existing_acoustic_nodes=num_existing_acoustic_nodes
                                                )
                    





   


os.makedirs('accuracy', exist_ok=True)

dataset = args.dataset
os.makedirs(f'accuracy/{dataset}', exist_ok=True)

# CSV file path
csv_file = f'{dataset}/accuracy_induc_val_{args.sub_units}_{args.drop_freq}_{args.drop_int}.csv'

# Check if the CSV file exists
file_exists = os.path.isfile(f'accuracy/{csv_file}')

# Create CSV file and write header if it does not exist
if not file_exists:
    with open(f'accuracy/{csv_file}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Heterogeneous regressor Model','Link prediction','link prediction topk','Heterogeneous sage Model', 'twa', 'num_n_h', 'mhg', 'num_n_a', 'ta', 'alpha', 'tw', 'msw', 'msa', 'mgw', 'mma', 'add','k_out', 'k_inf', 'lambda', 'density'])





# Train and evaluate SVM for heterogeneous regressor model embeddings
logging.info(f'Train and evaluate SVM for heterogeneous regressor model embeddings')

accuracy_hetero_regressor = train_evaluate_dnn_pytorch(X_train=acoustic_embeddings_regressor, X_test=acoustic_val_embeddings_regressor, y_train=labels_np, y_test=val_labels_np)
logging.info(f"Accuracy of the Heterogeneous regressor Model: {accuracy_hetero_regressor:.4f}")

logging.info(f'Link predictor accuracy')

with open(f'label_reencoder_{args.dataset}.pkl', 'rb') as f:   # Load all the labels names
    reencoder = pickle.load(f)

top_k = 3
acc_pred_link, acc_topk  = evaluate_acoustic_word_link_prediction_topk(
    gcn_model=model_hetero_regressor,
    hetero_graph=hetero_regressor_graph,
    num_existing_acoustic_nodes=num_existing_acoustic_nodes,
    labels_acoustic=val_labels_np,
    labels_word=hetero_graph.nodes['word'].data['label'],
    encoder = reencoder,
    top_k=top_k
)

acc_class_pred_link, acc_class_topk  = evaluate_link_prediction_classification_topk(
    gcn_model=model_hetero_regressor,
    hetero_graph=hetero_regressor_graph,
    num_existing_acoustic_nodes=num_existing_acoustic_nodes,
    labels_acoustic=val_labels_np,
    labels_word=hetero_graph.nodes['word'].data['label'],
    encoder = reencoder,
    top_k=top_k
)

logging.info(f"Link prediction accuracy: {acc_pred_link:.4f}")
logging.info(f"Link prediction {top_k}: {acc_topk:.4f}")

logging.info(f"Link class prediction accuracy: {acc_class_pred_link:.4f}")
logging.info(f"Link class prediction {top_k}: {acc_topk:.4f}")



# Write accuracy results to CSV file
logging.info(f'Write accuracy results to CSV file')
with open(f'accuracy/{csv_file}', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([ accuracy_hetero_regressor,acc_class_pred_link, acc_class_topk,acc_pred_link,acc_topk, float(args.twa), float(args.num_n_h), args.mhg, float(args.num_n_a), float(args.ta), float(args.alpha), float(args.tw), args.msw, args.msa, args.mgw, args.mma, args.add, args.k_out, args.k_inf, args.lamb, args.density])

