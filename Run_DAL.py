### Deep Active Learning - ARL For Graph Classification

#By Smayan Khanna

#Code help and credit from: https://github.com/ej0cl6/deep-active-learning
#This code is cited in the report/paper as well.

# General imports
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
import torch

import torch.nn.functional as F
from torch.utils.data import random_split
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
from torch_geometric.utils import to_networkx

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from sklearn.cluster import KMeans
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -----------------------
# Set random seed
# -----------------------

def set_seed(seed: int):
    """
    A simple function to set the random seed for reproducibility for Pytroch, NumPy, and Python etc.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# split manager
# -----------------------

class SplitManager:
    """
    A class to manage the splitting of a dataset into labeled, unlabeled, and test sets.
    """
    def __init__(self, dataset, init_idxs, test_idxs, batch_size=8, seed=42):
        self.ds = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.labeled = list(init_idxs)
        self.unlabeled = [i for i in range(len(dataset)) if i not in init_idxs + test_idxs]
        self.test = list(test_idxs)

    def get_loader(self, idxs, shuffle=False, batch=True):
        gen = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            Subset(self.ds, idxs),
            batch_size=self.batch_size if batch else 1,
            shuffle=shuffle,
            generator=gen if shuffle else None
        )

    def labeled_loader(self):
        return self.get_loader(self.labeled, shuffle=True)

    def unlabeled_loader(self):
        return self.get_loader(self.unlabeled)

    def test_loader(self):
        return self.get_loader(self.test, batch=False)

    def add_labels(self, new_idxs):
        self.labeled += new_idxs
        self.unlabeled = [i for i in self.unlabeled if i not in new_idxs]

    def get_dataset_subset(self, idxs):
        return Subset(self.ds, idxs)

# -----------------------
# MDOLES
# -----------------------

class GCN(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers=2):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = num_features if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_c, hidden_channels))
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def get_graph_embedding(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return global_mean_pool(x, batch)

class GAT(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers=2):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = num_features if i == 0 else hidden_channels
            self.convs.append(GATConv(in_c, hidden_channels))
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def get_graph_embedding(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return global_mean_pool(x, batch)

class SAGE(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers=2):
        super(SAGE, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = num_features if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_c, hidden_channels))
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def get_graph_embedding(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return global_mean_pool(x, batch)

class GIN(nn.Module):
    def __init__(self, hidden_channels, num_features, num_classes, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = num_features if i == 0 else hidden_channels
            
            # GIN uses a MLP per layer; here us 2-layer MLP with ReLU
            mlp = nn.Sequential(
                Linear(in_c, hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
            self.convs.append(GINConv(mlp))
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)

    def get_graph_embedding(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return global_mean_pool(x, batch)

# -----------------------
# Training / Eval / Embedding
# -----------------------

def train(model, optimizer, loader, num_epochs):
    model.train()
    total_loss = 0
    for _ in range(num_epochs):
        for data in loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct, total_loss = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    return correct / len(loader.dataset), total_loss / len(loader.dataset)


def compute_mean_embedding(model, loader):
    model.eval()
    embs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            emb = model.get_graph_embedding(data.x, data.edge_index, data.batch)
            embs.append(emb.cpu())
    embs = torch.cat(embs, dim=0)
    return embs.mean(dim=0)

# -----------------------
# Acquisition Functions
# -----------------------

def random_acquisition(unlabeled_set, k=10, model=None):
    """
    Returns the indices of k random samples from the unlabeled set.
    """
    # Randomly select k samples from the unlabeled set
    indices = np.random.choice(len(unlabeled_set), size=k, replace=False)

    return np.ndarray.tolist(indices)

def entropy_sampling(unlabeled_loader, model, k = 10, iter = 1):
    #Code credit: See top of file
    """
    Returns the indices of k samples from the unlabeled set based on entropy sampling. 
    By setting iter to 1 we are essentially just doing regular entropy sampling w/o dropout.
    """
    
    model.eval()
    entropy_values = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for data in unlabeled_loader:
            data = data.to(device)
            probs = []
            for _ in range(iter):
                logits = model(data.x, data.edge_index, data.batch) 
                probabilities = softmax(logits)
                probs.append(probabilities)
            
            probs = torch.stack(probs, dim=0)
            probs = torch.mean(probs, dim=0)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropy = entropy.cpu().tolist()
            entropy_values.extend(entropy)        
            
    values, indices = torch.topk(torch.tensor(entropy_values), k)
    return torch.Tensor.tolist(indices)

def embedding_sampling(unlabeled_loader, labelled_mean_embeddings, model, k = 10):
    """
    Returns the indices of k samples from the unlabeled set based on embedding sampling
    """
 
    embedding_cosine_similarity = []
    model.eval()

    with torch.no_grad():
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        for data in unlabeled_loader:
            data = data.to(device)
            embeddings = model.get_graph_embedding(data.x, data.edge_index, data.batch)
            labelled_mean_embeddings = labelled_mean_embeddings.to(embeddings.device)
            repeated_mean = labelled_mean_embeddings.unsqueeze(0).expand(embeddings.size(0), -1)

            distances = cosine_similarity(embeddings, repeated_mean)
            distances = distances.cpu().tolist()
            embedding_cosine_similarity.extend(distances)  
    
    values, indices = torch.topk(torch.tensor(embedding_cosine_similarity), k, largest=True)
    return torch.Tensor.tolist(indices)

def kmeans_sampling(loader, model, k):
    #Code credit: See top of file
    model.eval()
    embs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            embs.append(model.get_graph_embedding(data.x, data.edge_index, data.batch).cpu())
    embs = torch.cat(embs, dim=0).numpy()
    km = KMeans(n_clusters=k)
    km.fit(embs)
    cluster_idxs = km.predict(embs)
    centers = km.cluster_centers_[cluster_idxs]
    dist = ((embs - centers)**2).sum(axis=1)
    selected = []
    for i in range(k):
        pts = np.where(cluster_idxs==i)[0]
        selected.append(int(pts[dist[pts].argmin()]))
    return selected

def hybrid_sampling(unlabeled_loader, model, k, alpha=0.5, iter=1):
    """
    alpha ∈ [0,1] trades off uncertainty vs. diversity:
      score = alpha * normalized_entropy + (1-alpha) * normalized_diversity
    """
    model.eval()
    embs, entropies = [], []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for data in unlabeled_loader:
            data = data.to(device)
            #embeddings
            emb = model.get_graph_embedding(data.x, data.edge_index, data.batch)
            embs.append(emb.cpu())
            #collect entropies
            logits = model(data.x, data.edge_index, data.batch)
            probs = softmax(logits)
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropies.extend(ent.cpu().tolist())

    embs = torch.cat(embs, dim=0).numpy()
    entropies = np.array(entropies)

    km = KMeans(n_clusters=k).fit(embs)
    centers = km.cluster_centers_
    assignments = km.predict(embs)
    distances = np.linalg.norm(embs - centers[assignments], axis=1)

    #Normalize both to [0,1]
    e_min, e_max = entropies.min(), entropies.max()
    d_min, d_max = distances.min(), distances.max()
    ent_norm = (entropies - e_min) / (e_max - e_min + 1e-8)
    dist_norm = (distances - d_min) / (d_max - d_min + 1e-8)

    #Combine
    scores = alpha * ent_norm + (1 - alpha) * dist_norm

    topk_idxs = np.argsort(-scores)[:k]
    return topk_idxs.tolist()

# -----------------------
# Active Learning Loop
# -----------------------

def run_active_learning_loop(
    model_class, dataset, split_manager, seed,
    acquisition_strategy, k, n_rounds,
    hidden_channels, num_layers, num_epochs,
    device, reinitialize_model=False, batch_size=8, alpha=0.5
):
    test_acc, test_loss = [], []
    total_gradient_steps = 0

    if not reinitialize_model:
        set_seed(seed)
        model = model_class(hidden_channels, dataset.num_node_features,
                            dataset.num_classes, num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for round_idx in range(n_rounds):
        print(f"[Seed {seed}] Round {round_idx+1}/{n_rounds} | Strategy: {acquisition_strategy}")
        if reinitialize_model:
            set_seed(seed)
            model = model_class(hidden_channels, dataset.num_node_features,
                                dataset.num_classes, num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        l_loader = split_manager.labeled_loader()
        u_loader = split_manager.unlabeled_loader()
        t_loader = split_manager.test_loader()

        # Track gradient steps
        num_batches = int(np.ceil(len(l_loader.dataset) / batch_size))
        total_gradient_steps += num_epochs * num_batches

        print(f"Number of labeled samples: {len(l_loader.dataset)}")

        train(model, optimizer, l_loader, num_epochs)
        acc, loss = evaluate(model, t_loader)
        test_acc.append(acc)
        test_loss.append(loss)

        if acquisition_strategy == "random":
            idxs = random_acquisition(split_manager.get_dataset_subset(split_manager.unlabeled), k=k)
        elif acquisition_strategy == "entropy":
            idxs = entropy_sampling(u_loader, model, k)
        elif acquisition_strategy == "embedding":
            mean_emb = compute_mean_embedding(model, l_loader)
            idxs = embedding_sampling(u_loader, mean_emb, model, k)
        elif acquisition_strategy == "kmeans":
            idxs = kmeans_sampling(u_loader, model, k)
        elif acquisition_strategy == "hybrid":
            idxs = hybrid_sampling(u_loader, model, k, alpha=alpha)
        else:
            raise ValueError(f"Unknown strategy {acquisition_strategy}")

        global_idxs = [split_manager.unlabeled[i] for i in idxs]
        split_manager.add_labels(global_idxs)

    return test_acc, test_loss, total_gradient_steps

# -----------------------
# Argparse
# -----------------------

def create_split_manager(dataset, seed, test_frac=0.3, init_frac=0.05, batch_size=8):
    set_seed(seed)
    perm = torch.randperm(len(dataset)).tolist()
    n_total = len(dataset)
    n_test = int(test_frac * n_total)
    n_init = int(init_frac * n_total)
    return SplitManager(dataset, perm[:n_init], perm[n_total-n_test:], batch_size, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DD")
    parser.add_argument("--models", nargs='+', default=["GCN","GAT","SAGE"])
    parser.add_argument("--strategies", nargs='+', default=["random","entropy","embedding","kmeans"])
    parser.add_argument("--seeds", nargs='+', type=int, default=[0,1,2])
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[20])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_frac", type=float, default=0.3)
    parser.add_argument("--init_frac", type=float, default=0.1)
    # parser.add_argument("--reinit", action='store_true')
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = TUDataset(root="/project/vitelli/for_smayan/Deep_Active_Learning/data/TUDataset", name=args.dataset)
    model_map = {"GCN": GCN, "GAT": GAT, "SAGE": SAGE, "GIN": GIN}

    for m in args.models:
        for strat in args.strategies:
            for seed in args.seeds:
                for h in args.hidden_dims:
                    sm = create_split_manager(dataset, seed, batch_size=args.batch_size, test_frac=args.test_frac, init_frac=args.init_frac)
                    # set_seed(seed)
                    acc, loss, no_gradient_steps = run_active_learning_loop(
                        model_map[m], dataset, sm, seed,
                        strat, args.k, args.n_rounds,
                        h, args.num_layers, args.num_epochs, device
                    )
                    np.save(os.path.join(args.output_dir, f"{m}_{strat}_seed{seed}_hdim{h}_acc.npy"), np.array(acc))
                    np.save(os.path.join(args.output_dir, f"{m}_{strat}_seed{seed}_hdim{h}_loss.npy"), np.array(loss))
                    np.save(os.path.join(args.output_dir, f"{m}_{strat}_seed{seed}_hdim{h}_steps.npy"), np.array(no_gradient_steps))
                    print(f"No of gradient steps: {no_gradient_steps}")
                    print(f"Saved results for {m}, {strat}, seed={seed}, hdim={h}")
