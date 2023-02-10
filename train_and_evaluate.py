#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import os

import numpy as np

import torch as t
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from captum.attr import IntegratedGradients
from torch_geometric.nn import to_captum_model, to_captum_input
import tqdm

import dataset
import model

t.manual_seed(0)
np.random.seed(0)


def evaluate(mdl: model.GCN) -> float:
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
    return mse_test / mse_null


def train(mdl: model.GCN) -> model.GCN:
    mdl.train()
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = t.nn.MSELoss()

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
            loss = criterion(out, data.y[:, 0].reshape(-1, 1))
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

    mdl = model.GCN()
    mdl.load_state_dict(t.load(os.path.join("tmp", "mdl-age.ckpt")))
    # mdl = train(mdl)
    # t.save(mdl.state_dict(), os.path.join("tmp", "mdl-age.ckpt"))

    output_idx = 0
    batch_test = next(
        iter(DataLoader(dataset.data_test, batch_size=len(dataset.test_ids)))
    )

    explainer = Explainer(
        model=mdl,
        algorithm=GNNExplainer(epochs=200),
        explainer_config=dict(
            explanation_type="model",
            node_mask_type="object"
        ),
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    explanation = explainer.get_prediction(
        batch_test.x,
        batch_test.edge_index,
        batch_test.edge_attr,
        batch_test.batch,
        batch_test.y[:, 1:],
    )

    print(explanation)

""" output:
0.9614568
main() executed in 220.39 seconds
"""
