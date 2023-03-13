#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import datetime
import os

import numpy as np
import sklearn.metrics as skl_mets

import torch as t
from torch_geometric.loader import DataLoader
import tqdm

import dataset
import model

t.manual_seed(0)


def evaluate(
    mdl: model.GCN, return_preds=False
) -> float | tuple[float, dict[str, np.array]]:
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
    auc_test = skl_mets.roc_auc_score(ytest, ypred)
    mdl.train()
    if return_preds:
        return auc_test, {"preds": ypred, "trues": ytest}
    else:
        return auc_test


def train(mdl: model.GCN) -> model.GCN:
    mdl.train()
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in tqdm.tqdm(range(10)):
        print(evaluate(mdl))
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
            loss = t.nn.functional.binary_cross_entropy(
                out, data.y[:, 0].reshape(-1, 1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return mdl


if __name__ == "__main__":
    # import time

    # print contents of file to screen to remember what settings used when
    # running multiple versions simultaneously
    # with open(__file__) as f:
    #     print(f.read())
    #     print("%" * 79)
    # with open(model.__file__) as f:
    #     print(f.read())
    #     print("%" * 79)

    # t0 = time.time()

    mdl = model.GCN()
    mdl = train(mdl)
    t.save(
        mdl.state_dict(),
        os.path.join(
            "tmp",
            "mdl-sex-"
            + datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%MZ"
            )
            + ".ckpt",
        ),
    )
    # mdl.load_state_dict(
    #     t.load(os.path.join("tmp", "mdl-age-20230313T1421Z.ckpt"))
    # )

    auc, pred_true_dict = evaluate(mdl, return_preds=True)
    ytrue, ypred = pred_true_dict["trues"], pred_true_dict["preds"]

    print("auc: {auc:.2f}".format(auc=skl_mets.roc_auc_score(ytrue, ypred)))
    print(
        "acc: {acc:.2f}".format(
            acc=skl_mets.accuracy_score(ytrue, (ypred > 0.5).astype(int))
        )
    )
    print(
        "f1:  {f1:.2f}".format(
            f1=skl_mets.f1_score(ytrue, (ypred > 0.5).astype(int))
        )
    )

    # print(f"executed in {time.time()-t0:.2f} seconds")

""" output:
auc: 0.89
acc: 0.81
f1:  0.82
"""
