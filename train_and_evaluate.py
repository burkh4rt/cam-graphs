#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import numpy as np

import torch as t
from torch_geometric.loader import DataLoader
import tqdm

import dataset
import model

t.manual_seed(0)


def main():
    mdl = model.GCN()
    mdl.train()
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = t.nn.MSELoss()

    for _ in tqdm.tqdm(range(100)):
        loader_train = DataLoader(
            dataset.data_train, batch_size=700, shuffle=True
        )
        for data in iter(loader_train):
            out = mdl(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            print(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    mdl.eval()
    loader_test = DataLoader(
        dataset.data_test, batch_size=len(dataset.test_ids)
    )
    batch_test = next(iter(loader_test))
    preds_test = mdl(
        batch_test.x,
        batch_test.edge_index,
        batch_test.edge_attr,
        batch_test.batch,
    )

    mse_test = np.mean(
        np.square(preds_test.detach().numpy() - batch_test.y.numpy())
    )
    mse_null = np.mean(np.square(dataset.mean_train - batch_test.y.numpy()))
    print(f"{mse_test=:.2f}")
    print(f"{mse_null=:.2f}")
    print(f"{mse_test/mse_null=:.2f}")
    return preds_test.detach().numpy(), batch_test.y.numpy()


if __name__ == "__main__":
    import time

    t0 = time.time()
    ypred, ytest = main()
    print(f"main() executed in {time.time()-t0:.2f} seconds")

""" output:
mse_test=4.31
mse_null=4.46
mse_test/mse_null=0.97
main() executed in 21.46 seconds
"""
