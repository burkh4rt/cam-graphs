#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import datetime
import os

import numpy as np

import torch as t
from torch_geometric.loader import DataLoader
import tqdm

import dataset
import model

t.manual_seed(0)


def evaluate(mdl: model.GCN, return_preds=False) -> float:
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
    mdl.train()
    if return_preds:
        return mse_test / mse_null, {"preds": ypred, "trues": ytest}
    else:
        return mse_test / mse_null


def train(mdl: model.GCN) -> model.GCN:
    mdl.train()
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = t.nn.MSELoss()

    for _ in tqdm.tqdm(range(30)):
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return mdl


def main():
    mdl = model.GCN()
    mdl = train(mdl)
    t.save(
        mdl.state_dict(),
        os.path.join(
            "tmp",
            "mdl-age-"
            + datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%MZ"
            )
            + ".ckpt",
        ),
    )
    print(evaluate(mdl))


if __name__ == "__main__":
    import time

    # print contents of file to screen to remember what settings used when
    # running multiple versions simultaneously
    # with open(__file__) as f:
    #     print(f.read())
    #     print("%" * 79)
    # with open(model.__file__) as f:
    #     print(f.read())
    #     print("%" * 79)

    # t0 = time.time()
    # main()
    # print(f"main() executed in {time.time()-t0:.2f} seconds")

    mdl = model.GCN()
    mdl.load_state_dict(
        t.load(os.path.join("tmp", "mdl-age-20230313T1421Z.ckpt"))
    )
    n_mse, pred_true_dict = evaluate(mdl, return_preds=True)
    ytrue, ypred = pred_true_dict["trues"], pred_true_dict["preds"]

    print("mse null:", np.mean(np.square(ytrue - dataset.mean_train)))
    print("mse ours:", np.mean(np.square(ytrue - ypred)))

""" output:
0.5501894
main() executed in 574.80 seconds
"""
