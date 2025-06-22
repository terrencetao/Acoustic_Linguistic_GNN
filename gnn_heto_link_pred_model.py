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
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IdentityConv(nn.Module):
    def __init__(self, out_dim):
        super(IdentityConv, self).__init__()
        self.out_dim = out_dim

    def forward(self, graph, feat, edge_weight=None):
        if isinstance(feat, tuple):
            # On prend les features de destination (feat_dst)
            return feat[1][:, :self.out_dim]
        else:
            return feat[:, :self.out_dim]

class HeteroLinkGCN(nn.Module):
    def __init__(self, in_feats, hidden_size,nombre_phon, linear_hidden_size):
        super(HeteroLinkGCN, self).__init__()

        self.conv1 = HeteroGraphConv({
            'sim_tic': SAGEConv(in_feats['acoustic'], nombre_phon, 'mean'),
            'sim_w': IdentityConv(out_dim=nombre_phon),
            'related_to': SAGEConv(in_feats['acoustic'], nombre_phon, 'mean')
        }, aggregate='mean')

        self.conv2 = HeteroGraphConv({
            'sim_tic': SAGEConv(nombre_phon, nombre_phon, 'mean'),
            'sim_w': IdentityConv(out_dim=nombre_phon),
            'related_to': SAGEConv(nombre_phon, nombre_phon, 'mean')
        }, aggregate='mean')

        # MLP for binary classification (link exists or not)
        self.edge_predictor = nn.Sequential(
            nn.Linear(4 * nombre_phon, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, 1),
            nn.Sigmoid()  # Sortie entre 0 et 1
        )

    def forward(self, g, inputs):
        edge_weights = {etype: g.edges[etype].data['weight'] for etype in g.etypes}
        h = self.conv1(g, inputs, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        h = {k: F.relu(v) for k, v in h.items()}
        #h = self.conv2(g, h, mod_kwargs={k: {'edge_weight': v} for k, v in edge_weights.items()})
        return h  # embeddings

  


def train_link_prediction(model, g, features, true_edge_labels, src, dst, adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word, epochs=100, lr=0.0001, lamb=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCELoss()  # puisque on applique sigmoid dans le modèle

    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)
        pred_probs = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'], src, dst)

        classification_loss = criterion(pred_probs, true_edge_labels)
        topo_loss = topological_loss(embeddings['acoustic'], embeddings['word'], adj_matrix_acoustic, adj_matrix_word, adj_matrix_acoustic_word)
        loss =classification_loss +   lamb *topo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step(classification_loss)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, BCE Loss: {classification_loss.item():.4f}")

    return model
    
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

def train_link_regression2(model, g, features, true_edge_labels,
                          src, dst, src_pos, dst_pos,
                          val_ratio=0.1, test_ratio=0.1, epochs=100,
                          lr=0.0001, lamb=0.0, seed=42):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCELoss()

    # === Split train / val / test de manière reproductible ===
    num_edges = src.shape[0]
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_edges, generator=generator)

    test_size = int(test_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    train_size = num_edges - test_size - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_src, train_dst = src[train_idx], dst[train_idx]
    val_src, val_dst = src[val_idx], dst[val_idx]
    test_src, test_dst = src[test_idx], dst[test_idx]

    train_labels = true_edge_labels[train_idx]
    val_labels = true_edge_labels[val_idx]
    test_labels = true_edge_labels[test_idx]

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)

        # Training: Link prediction loss
        train_pred_probs = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                      train_src, train_dst)
        classification_loss = criterion(train_pred_probs, train_labels)

        # Contrastive loss
        contrast_loss = infoNCE_loss(embeddings['acoustic'], embeddings['word'],
                                     src_pos, dst_pos, temperature=0.07)

        # Total loss
        loss = classification_loss + alpha * contrast_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Validation ===
        model.eval()
        with torch.no_grad():
            val_pred_probs = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                        val_src, val_dst)
            val_loss = criterion(val_pred_probs, val_labels)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, "
                  f"BCE: {classification_loss.item():.4f}, "
                  f"Contrast: {contrast_loss.item():.4f}, "
                  f"Val BCE: {val_loss.item():.4f}")

    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # === Évaluation finale sur test ===
    model.eval()
    with torch.no_grad():
        embeddings = model(g, features)
        test_pred_probs = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                     test_src, test_dst)
        test_loss = criterion(test_pred_probs, test_labels)
        print(f"\nTest BCE Loss: {test_loss.item():.4f}")

    return model





import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def strong_link_detection_metric(preds, labels, threshold=0.8):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    mask = labels_np == 1.0
    if mask.sum() == 0:
        return {"count": 0, "correct": 0, "recall@0.8": None}
    correct = (preds_np[mask] >= threshold).sum()
    return {
        "count": int(mask.sum()),
        "correct": int(correct),
        "recall@0.8": correct / mask.sum()
    }


def weak_link_detection_metric(preds, labels, threshold=0.2):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    mask = labels_np == 0.0
    if mask.sum() == 0:
        return {"count": 0, "correct": 0, "recall@0.2": None}
    correct = (preds_np[mask] <= threshold).sum()
    return {
        "count": int(mask.sum()),
        "correct": int(correct),
        "recall@0.2": correct / mask.sum()
    }


def compute_metrics(preds, labels):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return {
        "MSE": mean_squared_error(labels_np, preds_np),
        "MAE": mean_absolute_error(labels_np, preds_np),
        "R2": r2_score(labels_np, preds_np)
    }


def train_link_regression(model, g, features, true_edge_labels,
                          src, dst, src_pos, dst_pos, 
                          val_ratio=0.2, test_ratio=0.1,
                          epochs=100, lr=0.0001, lamb=0.0, seed=42):
    
    device = next(model.parameters()).device  # Get device from model
    
    # Move all tensors to the same device as model
    g = g.to(device)
    features = {k: v.to(device) for k, v in features.items()}
    true_edge_labels = true_edge_labels.to(device)
    src = src.to(device)
    dst = dst.to(device)
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)
    

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCELoss()

    # === Train / Val / Test split ===
    num_edges = src.shape[0]
    indices = torch.randperm(num_edges)

    train_end = int((1 - val_ratio - test_ratio) * num_edges)
    val_end = int((1 - test_ratio) * num_edges)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_src, train_dst = src[train_idx], dst[train_idx]
    val_src, val_dst = src[val_idx], dst[val_idx]
    test_src, test_dst = src[test_idx], dst[test_idx]

    train_labels = true_edge_labels[train_idx]
    val_labels = true_edge_labels[val_idx]
    test_labels = true_edge_labels[test_idx]

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        embeddings = model(g, features)

        train_preds = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                 train_src, train_dst)
        classification_loss = criterion(train_preds, train_labels)

        contrast_loss = infoNCE_loss(embeddings['acoustic'], embeddings['word'],
                                     src_pos, dst_pos, temperature=0.07)

        loss = classification_loss + lamb * contrast_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       

       
                  
        # === Validation ===
        model.eval()
        with torch.no_grad():
            val_pred_probs = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                        val_src, val_dst)
            val_loss = criterion(val_pred_probs, val_labels)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()


        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, "
                  f"BCE: {classification_loss.item():.4f}, "
                  f"Contrast: {contrast_loss.item():.4f}, "
                  f"Val BCE: {val_loss.item():.4f}")


    # === Evaluation ===
    model.eval()
    with torch.no_grad():
        embeddings = model(g, features)

        train_preds = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                 train_src, train_dst)
        val_preds = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                               val_src, val_dst)
        test_preds = predict_edge_probabilities(model, embeddings['acoustic'], embeddings['word'],
                                                test_src, test_dst)

        train_metrics = compute_metrics(train_preds, train_labels)
        val_metrics = compute_metrics(val_preds, val_labels)
        test_metrics = compute_metrics(test_preds, test_labels)

        strong_train = strong_link_detection_metric(train_preds, train_labels)
        strong_val = strong_link_detection_metric(val_preds, val_labels)
        strong_test = strong_link_detection_metric(test_preds, test_labels)

        weak_train = weak_link_detection_metric(train_preds, train_labels)
        weak_val = weak_link_detection_metric(val_preds, val_labels)
        weak_test = weak_link_detection_metric(test_preds, test_labels)

        label_distribution = {
            "train": {
                "count_1.0": int((train_labels == 1.0).sum().item()),
                "count_0.0": int((train_labels == 0.0).sum().item())
            },
            "val": {
                "count_1.0": int((val_labels == 1.0).sum().item()),
                "count_0.0": int((val_labels == 0.0).sum().item())
            },
            "test": {
                "count_1.0": int((test_labels == 1.0).sum().item()),
                "count_0.0": int((test_labels == 0.0).sum().item())
            }
        }

    return {
        "model": model,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        },
        "strong_link_detection": {
            "train": strong_train,
            "val": strong_val,
            "test": strong_test
        },
        "weak_link_detection": {
            "train": weak_train,
            "val": weak_val,
            "test": weak_test
        },
        "label_distribution": label_distribution,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        }
    }

    
def contrastive_edge_loss(embeddings_acoustic, embeddings_word, pos_src, pos_dst, neg_src, neg_dst, margin=1.0):
    # Get positive pairs
    positive_word = embeddings_word[pos_src]
    positive_acoustic = embeddings_acoustic[pos_dst]
    pos_distance = F.pairwise_distance(positive_word, positive_acoustic)
    
    # Get negative pairs
    negative_word = embeddings_word[neg_src]
    negative_acoustic = embeddings_acoustic[neg_dst]
    neg_distance = F.pairwise_distance(negative_word, negative_acoustic)
    
    # Loss: ensure positive distance is smaller than negative distance by margin
    loss = torch.mean(torch.clamp(pos_distance - neg_distance + margin, min=0))
    return loss



def infoNCE_loss(embeddings_acoustic, embeddings_word, pos_src, pos_dst, temperature=0.07):
    """
    embeddings_acoustic: Tensor [N, D]
    embeddings_word: Tensor [N, D]
    pos_src, pos_dst: indices des paires positives (généralement identiques dans batch ordonné)
    temperature: facteur d'échelle pour la similarité
    """
    device = embeddings_acoustic.device
    pos_src = pos_src.to(device)
    pos_dst = pos_dst.to(device)
    #
    # Normaliser les embeddings pour utiliser cosinus similarity
    embeddings_acoustic = F.normalize(embeddings_acoustic, dim=1)
    embeddings_word = F.normalize(embeddings_word, dim=1)

    # Extraire les embeddings positifs
    positive_word = embeddings_word[pos_src]    # [batch_size, D]
    positive_acoustic = embeddings_acoustic[pos_dst]  # [batch_size, D]

    # Calculer la matrice de similarités entre chaque mot et chaque acoustique (cosinus)
    logits = torch.matmul(positive_word, positive_acoustic.T)  # [batch_size, batch_size]
    logits = logits / temperature

    # Les labels indiquent que chaque mot correspond au même index acoustique
    labels = torch.arange(logits.size(0)).to(logits.device)

    # Calculer la cross entropy (InfoNCE)
    loss = F.cross_entropy(logits, labels)

    return loss

    
    
    
def topological_loss(embeddings_acoustic, embeddings_word,
                     adj_matrix_acoustic, adj_matrix_word, adj_matrix_word_acoustic,
                     lamb_1=0.0, lamb_2=1):
    # Acoustic similarity (N_acoustic x N_acoustic)
    cosine_sim_acoustic = F.cosine_similarity(
        embeddings_acoustic.unsqueeze(1), embeddings_acoustic.unsqueeze(0), dim=2)
    cosine_sim_acoustic = cosine_sim_acoustic - torch.diag_embed(torch.diag(cosine_sim_acoustic))
    loss_acoustic = F.mse_loss(cosine_sim_acoustic, adj_matrix_acoustic)

    # Word similarity (N_word x N_word)
    cosine_sim_word = F.cosine_similarity(
        embeddings_word.unsqueeze(1), embeddings_word.unsqueeze(0), dim=2)
    cosine_sim_word = cosine_sim_word - torch.diag_embed(torch.diag(cosine_sim_word))
    loss_word = F.mse_loss(cosine_sim_word, adj_matrix_word)
    
    # Acoustic-Word similarity (N_word x N_acoustic)
    cosine_sim_word_acoustic = F.cosine_similarity(
        embeddings_word.unsqueeze(1),               # [N_word, 1, D]
        embeddings_acoustic.unsqueeze(0),           # [1, N_acoustic, D]
        dim=2                                       # compare along D
    )
    loss_acoustic_word = F.mse_loss(cosine_sim_word_acoustic, adj_matrix_word_acoustic)

    # Total loss
    return loss_acoustic + lamb_1 * loss_word + lamb_2 * loss_acoustic_word
    
    

def generate_negative_edges(g, num_samples, src_type='word', dst_type='acoustic', etype='related_to'):
    device = g.device
    src_nodes = torch.arange(g.num_nodes(src_type))
    dst_nodes = torch.arange(g.num_nodes(dst_type))
    existing_edges = set(zip(*g.edges(etype=etype)))
    
    negatives = []
    while len(negatives) < num_samples:
        src = torch.randint(0, g.num_nodes(src_type), (1,)).item()
        dst = torch.randint(0, g.num_nodes(dst_type), (1,)).item()
        if (src, dst) not in existing_edges:
            negatives.append((src, dst))
            existing_edges.add((src, dst))  # pour éviter les doublons
    return torch.tensor([i[0] for i in negatives], device=device), torch.tensor([i[1] for i in negatives], device=device)

def predict_edge_probabilities(model, acoustic_embeddings, word_embeddings, src, dst):
    device = next(model.parameters()).device
    src = src.to(device)
    dst = dst.to(device)
    src_embed = word_embeddings[src]
    dst_embed = acoustic_embeddings[dst]
    diff = torch.abs(src_embed - dst_embed)
    prod = src_embed * dst_embed
    edge_features = torch.cat([src_embed, dst_embed, diff, prod], dim=1)

    predicted_probs = model.edge_predictor(edge_features).squeeze()
    return predicted_probs

def predict_edge_probabilities_dot(model, acoustic_embeddings, word_embeddings, src, dst):
    dst_embed = acoustic_embeddings[dst]
    src_embed = word_embeddings[src]

    score = (src_embed * dst_embed).sum(dim=1)  # Dot product
    prob = torch.sigmoid(score)
    return prob
    
    
import pandas as pd
import os

def save_results_to_excel(results_dict, args, excel_path='results.xlsx'):
    # Ajouter les paramètres d'arguments au dictionnaire
    args_dict = {
        'twa': args.twa,
       
        'mhg': args.mhg,
        
        'lamb': args.lamb,
        
        'ta': args.ta,
        'tw': args.tw,
        'msw': args.msw,
        'msa': args.msa,
        'mgw': args.mgw,
        'mma': args.mma,
        'sub_units': args.sub_units,
        'dataset': args.dataset,
        'feature': args.feature,
    }

    # Fusionner les métriques avec les paramètres
    full_result = {**args_dict, **results_dict}

    # Charger ou créer le fichier Excel
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame()

    # Ajouter la nouvelle ligne
    df = pd.concat([df, pd.DataFrame([full_result])], ignore_index=True)

    # Enregistrer
    df.to_excel(excel_path, index=False)

    
def main(input_folder, graph_file, epochs, lamb, dataset, args):
    # Detect GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    glist, _ = dgl.load_graphs(os.path.join(input_folder, graph_file))
    hetero_graph = glist[0].to(device)  # Move graph to GPU
    
    features = {
        'acoustic': hetero_graph.nodes['acoustic'].data['feat'].to(device),
        'word': hetero_graph.nodes['word'].data['feat'].to(device)
    }   
    with open(f'phon_idx_{dataset}.pkl', 'rb') as f:
        phon_idx = pickle.load(f)
    
    # Get edge data
    # Positifs
    src_pos, dst_pos = hetero_graph.edges(etype=('word', 'related_to', 'acoustic'))
    weights_pos = hetero_graph.edges[('word', 'related_to', 'acoustic')].data['weight']
    
    # Move these to GPU
    src_pos = src_pos.to(device)
    dst_pos = dst_pos.to(device)
    weights_pos = weights_pos.to(device)


    

    # Négatifs (autant que les positifs, ou plus/moins si tu veux)
    src_neg, dst_neg = generate_negative_edges(hetero_graph, num_samples=len(src_pos))
    weights_neg = torch.zeros(len(src_neg))

    # Fusionne
    src = torch.cat([src_pos, src_neg], dim=0)
    dst = torch.cat([dst_pos, dst_neg], dim=0)
    true_edge_weights = torch.cat([weights_pos, weights_neg], dim=0)
    perm = torch.randperm(len(src))
    src, dst, true_edge_labels = src[perm], dst[perm], true_edge_weights[perm]  # shuffle

    # Graph meta info
    in_feats = {ntype: features[ntype].shape[1] for ntype in features}
    hidden_size = 512
    linear_hidden_size = 64
    nb_phon = hetero_graph.nodes['word'].data['feat'].shape[1]
    model = HeteroLinkGCN(in_feats, hidden_size, nb_phon, linear_hidden_size).to(device)

   
   
    num_acoustic = hetero_graph.num_nodes('acoustic')
    num_word = hetero_graph.num_nodes('word')

    edge_weights = hetero_graph.edges[('word', 'related_to', 'acoustic')].data['weight']
    
    features = {k: v.float() for k, v in features.items()}
    true_edge_weights = true_edge_weights.float()

    train_test_dic = train_link_regression(
    model, hetero_graph, features, true_edge_labels,
    src, dst,
    src_pos, dst_pos,
    epochs=epochs, lr=0.0001, lamb=0.01  # adjust lamb/alpha as needed
)

    

    model =  train_test_dic['model']
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', "hetero_gnn_edge_regressor.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')
    
    os.makedirs(f'resultats/{dataset}', exist_ok=True)
    result_path = f'resultats/{dataset}/resultat.xlsx'
    # Enregistrer les résultats dans Excel
    save_results_to_excel(train_test_dic, args, result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='source folder', required=True)
    parser.add_argument('--graph_file', help='graph for training', required=True)
    parser.add_argument('--lamb', help='hyperparameter for objective function', required=True)
    parser.add_argument('--epochs', help='number of epochs', required=True)
    parser.add_argument('--dataset', help='name of dataset', required=True)
    parser.add_argument('--twa', help='word similarity threshold', required=True)
   
    parser.add_argument('--mhg', help='method to compute a word similarity', required=True)

    

    parser.add_argument('--ta', help='method to compute a word similarity', required=True)

    parser.add_argument('--tw', help='method to compute a word similarity', required=True)
    parser.add_argument('--msw', help='method to compute a word similarity', required=True)
    parser.add_argument('--msa', help='method to compute heterogeneous similarity', required=True)
    parser.add_argument('--mgw', help='method to build word graph ', required=True)
    parser.add_argument('--mma', help='method to build acoustic matrix', required=True)

    parser.add_argument('--sub_units', help='fraction of data', required=True)  
    parser.add_argument('--feature', type=str, default='mfcc', choices=['mfcc', 'mel_spec', 'wav2vec', 'trill', 'vggish', 'yamnet', 'wavlm', 'hubert'], help='Feature type to extract')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU.")
    

    main(args.input_folder, args.graph_file, int(args.epochs), float(args.lamb), args.dataset, args)




