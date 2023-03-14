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
    def __init__(
        self,
        alpha_dropout=0.05,
        gat_heads=3,
        gat_out_channels=3,
        dim_penultimate=5,
    ):
        super().__init__()
        self.alpha_dropout = alpha_dropout
        self.gat_heads = gat_heads
        self.gat_out_channels = gat_out_channels
        self.dim_penultimate = dim_penultimate
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
            self.dim_penultimate,
        )
        self.bnorm2 = BatchNorm1d(self.dim_penultimate)
        self.lin2 = Linear(self.dim_penultimate, 1)

    def forward(self, x, edge_index, edge_attr, batch, graph_feats):
        x1 = t.tanh(self.conv1(x, edge_index, edge_attr))
        x1 = self.agg(x1, batch)
        x1 = self.bnorm1(x1)
        x = to_dense_batch(x, batch, max_num_nodes=dataset.num_nodes)[0]
        x = x.view(-1, x.size(1) * x.size(2))
        x = t.cat((x1, x, graph_feats), -1)
        x = AlphaDropout(p=self.alpha_dropout)(x)
        x = t.tanh(self.lin1(x))
        x = self.bnorm2(x)
        x = self.lin2(x)
        return dataset.std_train * x + dataset.mean_train


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    self = GCN(
        alpha_dropout=0.1, gat_heads=4, gat_out_channels=4, dim_penultimate=4
    )
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
  (conv1): GATConv(1, 4, heads=4)
  (agg): MultiAggregation([
    MeanAggregation(),
    StdAggregation(),
    SoftmaxAggregation(learn=True),
  ], mode=cat)
  (bnorm1): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lin1): Linear(in_features=117, out_features=4, bias=True)
  (bnorm2): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lin2): Linear(in_features=4, out_features=1, bias=True)
)
forward function works
"""
