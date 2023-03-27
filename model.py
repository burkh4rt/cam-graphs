#!/usr/bin/env python3

"""
Defines a simple graph-based model with convolution and readout layers
"""

import numpy as np
import torch as t
from torch.nn import Linear, AlphaDropout, BatchNorm1d
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import aggr

import dataset

dataset = dataset.dataset(0)


class GCN(t.nn.Module):
    def __init__(
        self,
        alpha_dropout=0.05,
        gat_heads=3,
        gat_out_channels=3,
        dim_penultimate=5,
        mean_train=dataset.mean_train,
        std_train=dataset.std_train,
    ):
        super().__init__()
        self.mean_train = mean_train
        self.std_train = std_train
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
        self.alpha_dropout = alpha_dropout
        self.a_dropout_layer = AlphaDropout(p=self.alpha_dropout)
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
        x = self.a_dropout_layer(x)
        x = t.tanh(self.lin1(x))
        x = self.bnorm2(x)
        x = self.lin2(x)
        return self.std_train * x + self.mean_train

    def as_function_of_x_y(self, x_y1):
        self.eval()
        n = x_y1.shape[0]
        outp = np.zeros(n).reshape(n, 1)
        for i in range(n):
            x = x_y1[i, : -dataset.num_graph_features].reshape(
                dataset.batch_0.x.detach().numpy().shape
            )
            y = x_y1[i, -dataset.num_graph_features :].reshape(
                dataset.batch_0.y[:, 1:].detach().numpy().shape
            )
            outp[i] = (
                self.forward(
                    t.tensor(x, dtype=t.float),
                    dataset.batch_0.edge_index,
                    dataset.batch_0.edge_attr,
                    dataset.batch_0.batch,
                    t.tensor(y, dtype=t.float),
                )
                .detach()
                .numpy()
            )
        return outp


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    self = GCN(
        alpha_dropout=0.1, gat_heads=4, gat_out_channels=4, dim_penultimate=4
    )
    print(self)

    batch = next(iter(DataLoader(dataset.data_train, batch_size=8)))

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
