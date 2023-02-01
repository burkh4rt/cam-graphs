#!/usr/bin/env python3

"""
Defines a simple graph-based model with convolution and readout layers
"""

import torch as t
from torch.nn import Linear, AlphaDropout, BatchNorm1d
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
        self.bnorm = BatchNorm1d(71)
        self.lin1 = Linear(71, 25)
        self.lin2 = Linear(25, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = t.tanh(self.conv1(x, edge_index, edge_attr))
        x1 = self.agg(x1, batch)
        x = to_dense_batch(x, batch, max_num_nodes=dataset.num_nodes)[0]
        x = x.view(-1, x.size(1) * x.size(2))
        x = t.cat((x1, x), -1)
        x = self.bnorm(x)
        x = t.tanh(self.lin1(x))
        x = AlphaDropout(p=0.1)(x)
        x = self.lin2(x)
        return dataset.std_train * x + dataset.mean_train


if __name__ == "__main__":
    print(GCN())

""" output:
GCN(
  (conv1): GCNConv(1, 3)
  (agg): MultiAggregation([
    MeanAggregation(),
    StdAggregation(),
    SoftmaxAggregation(learn=True),
  ], mode=cat)
  (bnorm): BatchNorm1d(71, eps=1e-05, momentum=0.1, affine=True,
    track_running_stats=True)
  (lin1): Linear(in_features=71, out_features=25, bias=True)
  (lin2): Linear(in_features=25, out_features=1, bias=True)
)
"""
