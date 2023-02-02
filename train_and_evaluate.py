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
            dataset.data_train, 1000  # batch_size=len(dataset.train_ids)
        )
        for data in iter(loader_train):
            out = mdl(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
                data.y[:, 1:],
            )
            loss = criterion(out, data.y[:, 0].reshape(-1, 1))
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
        batch_test.y[:, 1:],
    )

    ypred = preds_test.detach().numpy().ravel()
    ytest = batch_test.y[:, 0].numpy().ravel()

    mse_test = np.mean(np.square(ypred - ytest))
    mse_null = np.mean(np.square(dataset.mean_train - ytest))
    print(f"{mse_test=:.2f}")
    print(f"{mse_null=:.2f}")
    print(f"{mse_test/mse_null=:.2f}")
    return ypred, ytest


if __name__ == "__main__":
    import time

    t0 = time.time()
    ypred, ytest = main()
    print(f"main() executed in {time.time()-t0:.2f} seconds")

""" output:
mse_test=3.98
mse_null=4.14
mse_test/mse_null=0.96
main() executed in 523.74 seconds
"""
