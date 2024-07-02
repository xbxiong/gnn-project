
import os
import argparse
from turtle import forward
from sklearn import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, ReLU, ParameterDict, Parameter
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, HeteroConv, Sequential, GATConv
from torch_geometric.utils import to_undirected


class RGCN(torch.nn.Module):
    def __init__(self, data, in_channels, hidden_channels, out_channels,
                num_layer, dropout, target_type, ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.conv = ModuleList()

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })
    def forward(self, x, ):
        pass

class RGAT(torch.nn.Module):
    def __init__(self, data, in_channels, hidden_channels, out_channels,
                num_layer, dropout, target_type, ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.conv = ModuleList()

        
        

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

def train(model):
    # Create embeddings for all node types that do not come with features.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    pass



def main(args):
    dataset = PygNodePropPredDataset(name='pgbn-mag')
    data = dataset[0]
    edge_index_dict = data.edge_index_dict

    # We need to add reverse edges to the heterogeneous graph.
    r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
    edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('author', 'writes', 'paper')]
    edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

    r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
    edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
    edge_index_dict[('paper', 'cites', 'paper')] = edge_index
    pass
    train_idx = data['paper']

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    model = RGCN(in_channels=128, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes,\
        num_layer=args.num_layers, dropout=args.dropout, target_type=args.label_class)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gat', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropedge', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--label_class', type=str, default='paper')
    parser.add_argument('--record_file', type=str, default='record.txt')
    args = parser.parse_args()
    main(args)