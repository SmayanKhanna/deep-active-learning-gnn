import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCN(nn.Module):
    def __init__(self, hidden_channels, in_feats, out_feats, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_feats if i==0 else hidden_channels
            self.convs.append(GCNConv(ch_in, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_feats)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class GAT(nn.Module):
    def __init__(self, hidden_channels, in_feats, out_feats, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_feats if i==0 else hidden_channels
            self.convs.append(GATConv(ch_in, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_feats)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class SAGE(nn.Module):
    def __init__(self, hidden_channels, in_feats, out_feats, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_feats if i==0 else hidden_channels
            self.convs.append(SAGEConv(ch_in, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_feats)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


def train_one(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def eval_one(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS', choices=['PROTEINS','DD','NCI1','ENZYMES'])
    parser.add_argument('--models', nargs='+', default=['GCN'], choices=['GCN','GAT','SAGE'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,17,73])
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test_frac', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root='data/TUDataset', name=args.dataset)

    model_map = {'GCN':GCN,'GAT':GAT,'SAGE':SAGE}
    results = {}

    for m in args.models:
        accs = []
        for seed in args.seeds:
            set_seed(seed)
            # split indices
            n = len(dataset)
            perm = torch.randperm(n).tolist()
            n_test = int(args.test_frac*n)
            train_idx = perm[:-n_test]
            test_idx  = perm[-n_test:]
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
            test_loader  = DataLoader(Subset(dataset, test_idx ), batch_size=args.batch_size, shuffle=False)

            # model init
            model = model_map[m](args.hidden, dataset.num_node_features, dataset.num_classes, args.layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # train
            for _ in range(args.epochs):
                train_one(model, train_loader, optimizer, device)

            # eval
            acc = eval_one(model, test_loader, device)
            accs.append(acc)
            print(f"{m} | Seed {seed} | Test Acc: {acc:.4f}")

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        results[m] = (mean_acc, std_acc)
        print(f"{m} | Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")

    #save to npy
    out = {m: {'mean':results[m][0], 'std':results[m][1]} for m in results}
    np.save(f"full_training_{args.dataset}.npy", out)
