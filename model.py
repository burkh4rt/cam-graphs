#!/usr/bin/env python3

"""
Defines a simple graph-based model with convolution and readout layers
"""

import torch as t
from torch.nn import Linear, AlphaDropout, BatchNorm1d
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import aggr

import dataset


class GCN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat_out_channels = 3
        self.gat_heads = 3
        self.conv1 = GATConv(
            dataset.num_node_features,
            self.gat_out_channels,
            heads=self.gat_heads,
            add_self_loops=False,  # we've already included these
            improved=True,
            bias=False,
        )
        self.agg = aggr.MultiAggregation(
            ["mean", "std", aggr.SoftmaxAggregation(learn=True)]
        )
        self.bnorm1 = BatchNorm1d(3 * self.gat_out_channels * self.gat_heads)
        self.lin1 = Linear(
            3 * self.gat_out_channels * self.gat_heads
            + dataset.num_node_features * dataset.num_nodes
            + dataset.num_graph_features,
            5,
        )
        self.bnorm2 = BatchNorm1d(5)
        self.lin2 = Linear(5, 1)

    def forward(self, x, edge_index, edge_attr, batch, graph_feats):
        x1 = t.tanh(self.conv1(x, edge_index, edge_attr))
        x1 = self.agg(x1, batch)
        x1 = self.bnorm1(x1)
        x = to_dense_batch(x, batch, max_num_nodes=dataset.num_nodes)[0]
        x = x.view(-1, x.size(1) * x.size(2))
        x = t.cat((x1, x, graph_feats), -1)
        x = AlphaDropout(p=0.05)(x)
        x = t.tanh(self.lin1(x))
        x = self.bnorm2(x)
        x = self.lin2(x)
        return dataset.std_train * x + dataset.mean_train


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    self = GCN()
    print(self)

    batch = next(
        iter(
            DataLoader(
                dataset.data_train, 1000  # batch_size=len(dataset.train_ids)
            )
        )
    )

    x, edge_index, edge_attr, batch, graph_feats = (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.y[:, 1:],
    )

    try:
        self.forward(x, edge_index, edge_attr, batch, graph_feats)
        print("forward function works")
    except Exception:
        print("forward function does not work")

""" output:
GCN(
  (conv1): GATConv(1, 3, heads=3)
  (agg): MultiAggregation([
    MeanAggregation(),
    StdAggregation(),
    SoftmaxAggregation(learn=True),
  ], mode=cat)
  (bnorm1): BatchNorm1d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lin1): Linear(in_features=97, out_features=5, bias=True)
  (bnorm2): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lin2): Linear(in_features=5, out_features=1, bias=True)
)
forward function works
"""
