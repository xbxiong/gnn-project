import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, HeteroConv, Sequential, GATConv

class RGCN(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_layers, dropout, tgt_type, embed):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.tgt_type = tgt_type
        in_channels = data[tgt_type]['x'].size(1) 
                    
        if embed:
            self.emb_dict = nn.ParameterDict({
                f'{k}': nn.Parameter(torch.Tensor(len(data[k]['x']), in_channels))
                for k in data.node_types if k != tgt_type
            })
        else:
            self.emb_dict = None
    
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            cin = cout = hidden_channels
            if _ == 0: cin = in_channels
            if _ == num_layers - 1: cout = out_channels

            conv = HeteroConv({
                k: SAGEConv((cin, cin), cout) for k in data.edge_types
            }, aggr='sum')
            self.convs.append(conv)
        
    def forward(self, x_dict, edge_index_dict):
        if self.emb_dict is not None:
            for k in x_dict:
                if k != self.tgt_type:
                    x_dict[k] = self.emb_dict[k][x_dict[k]]
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i != self.num_layers - 1:
                x_dict = {k: F.dropout(
                    F.relu(x), p=self.dropout, training=self.training) for k, x in x_dict.items()}
        return x_dict[self.tgt_type]
    
def train(loader, model, rank):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    total_examples = total_loss = 0
    batch_num = 0
    torch.cuda.cudart().cudaProfilerStart()
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(rank)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out[:batch_size], batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        
        if batch_num == 9 : torch.cuda.cudart().cudaProfilerStop()
        batch_num += 1

    return total_loss / total_examples

@torch.no_grad()
def test(loader, model, rank):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(rank)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples
    
def main_worker(rank, npros, dataset):
    world_size = npros
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13579'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    tgt_type = 'paper'
    data = dataset[0]
    
    embed = True
    for k in data.node_types:
        if k != tgt_type:
            data[k]['x'] = torch.arange(data[k]['num_nodes']) # one-hot embeddings
            
    train_idx = data['paper'].train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank].numpy() #[rank]
    sub_mask = np.zeros_like(data['paper'].train_mask)
    sub_mask[train_idx] = True
    sub_mask = torch.from_numpy(sub_mask)
    train_subinput_nodes = ('paper', sub_mask)
    
    train_loader = NeighborLoader(data, num_neighbors=[25, 20], batch_size=1024, shuffle=True,
                                  input_nodes=train_subinput_nodes, num_workers=12)
    
    model = RGCN(data, hidden_channels=64, out_channels=dataset.num_classes,
                 num_layers=2, dropout=0.5, tgt_type=tgt_type, embed=embed).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    for epoch in range(1, 4):
        loss = train(train_loader, model, rank)
            
        dist.barrier()
        
        if rank == 0:
            torch.cuda.empty_cache()
            val_input_nodes = ('paper', data['paper'].val_mask)
            val_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=256,
                                        input_nodes=val_input_nodes, num_workers = 6)
            
            val_acc = test(val_loader, model, rank)
                
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
            
    dist.barrier()
        
    if rank == 0:
        torch.cuda.empty_cache()
        test_input_nodes = ('paper', data['paper'].test_mask)
        test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=1024,
                                    input_nodes=test_input_nodes)
        test_acc = test(test_loader, model, rank)
        print('Test Acc: {:.4f}'.format(test_acc))
    
    torch.save(model.module.state_dict(), 'dist_rgcn1.pth')
    dist.destroy_process_group()
    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    path = './data'
    dataset = OGB_MAG(root=path, transform=T.ToUndirected())
    mp.spawn(main_worker, args=(world_size, dataset), nprocs=world_size, join=True)