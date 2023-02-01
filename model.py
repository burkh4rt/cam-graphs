#!/usr/bin/env python3

"""
Defines a simple graph-based model with convolution and readout layers
"""

import torch as t
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import aggr

import dataset


class GCN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            dataset.num_node_features,
            3,
            add_self_loops=False,  # we've already included these
            normalize=False,
            improved=True,
            bias=False,
        )
        self.agg = aggr.MultiAggregation(
            ["mean", "std", aggr.SoftmaxAggregation(learn=True)]
        )
        self.lin1 = Linear(310, 25)
        self.bnorm = t.nn.BatchNorm1d(25)
        self.lin2 = Linear(25, 10)
        self.lin3 = Linear(10, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.conv1(x, edge_index, edge_attr).relu()
        x1 = to_dense_batch(x1, batch, max_num_nodes=dataset.num_nodes)[0]
        x1 = x1.view(-1, x1.size(1) * x1.size(2))
        x = to_dense_batch(x, batch, max_num_nodes=dataset.num_nodes)[0]
        x = x.view(-1, x.size(1) * x.size(2))
        x = t.cat((x1, x), -1)
        x = Dropout(p=0.05)(x)
        x = self.lin1(x).relu()
        x = self.bnorm(x)
        x = self.lin2(x).relu()
        x = self.lin3(x)
        return x + dataset.mean_train
