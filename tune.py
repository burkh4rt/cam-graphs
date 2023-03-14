#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import datetime
import os

import torch as t
from torch_geometric.loader import DataLoader

import optuna as opt

import dataset
import model

t.manual_seed(0)


def objective(trial):
    mdl = model.GCN()
    optimizer = t.optim.Adagrad(
        mdl.parameters(),
        lr=trial.suggest_float("lr", 1e-3, 1e-2),
        weight_decay=trial.suggest_float("wd", 1e-4, 1e-3),
    )
    criterion = t.nn.MSELoss()

    for _ in range(10):
        loader_train = DataLoader(
            dataset.data_train, batch_size=1000, shuffle=True
        )
        for data in iter(loader_train):
            mdl.train()
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

    mdl.eval()
    batch_val = next(
        iter(
            DataLoader(
                dataset.data_val,
                batch_size=len(dataset.val_ids),
                shuffle=False,
            )
        )
    )
    loss_val = criterion(
        mdl(
            batch_val.x,
            batch_val.edge_index,
            batch_val.edge_attr,
            batch_val.batch,
            batch_val.y[:, 1:],
        ),
        batch_val.y[:, 0].reshape(-1, 1),
    )
    return loss_val


if __name__ == "__main__":
    study = opt.create_study()
    study.optimize(objective, n_trials=10)

    print(study.best_params)
    print(study.best_value)

"""
[I 2023-03-14 09:51:20,858] A new study created in memory with name: no-name-29e0c43b-95e1-4022-bc52-28f7d1bd0282
[I 2023-03-14 09:52:57,078] Trial 0 finished with value: 3.837623357772827 and parameters: {'lr': 0.005242891105447445, 'wd': 0.000689226751138232}. Best is trial 0 with value: 3.837623357772827.
[I 2023-03-14 09:54:33,204] Trial 1 finished with value: 4.6321797370910645 and parameters: {'lr': 0.0013485685583865428, 'wd': 0.00036968864808714045}. Best is trial 0 with value: 3.837623357772827.
[I 2023-03-14 09:56:11,988] Trial 2 finished with value: 3.6440277099609375 and parameters: {'lr': 0.007634639307970604, 'wd': 0.00041714770750167057}. Best is trial 2 with value: 3.6440277099609375.
[I 2023-03-14 09:57:48,970] Trial 3 finished with value: 3.8052637577056885 and parameters: {'lr': 0.0062839324680143685, 'wd': 0.000809992804792344}. Best is trial 2 with value: 3.6440277099609375.
[I 2023-03-14 09:59:27,484] Trial 4 finished with value: 3.6642394065856934 and parameters: {'lr': 0.009237045903937983, 'wd': 0.000541227441417108}. Best is trial 2 with value: 3.6440277099609375.
[I 2023-03-14 10:01:03,966] Trial 5 finished with value: 3.7190518379211426 and parameters: {'lr': 0.009693348834631452, 'wd': 0.0002264407895906013}. Best is trial 2 with value: 3.6440277099609375.
[I 2023-03-14 10:02:40,218] Trial 6 finished with value: 3.659353494644165 and parameters: {'lr': 0.008510449977221519, 'wd': 0.00027274458065503214}. Best is trial 2 with value: 3.6440277099609375.
[I 2023-03-14 10:04:16,771] Trial 7 finished with value: 3.631498098373413 and parameters: {'lr': 0.008819657239070081, 'wd': 0.0009277374148896613}. Best is trial 7 with value: 3.631498098373413.
[I 2023-03-14 10:05:52,138] Trial 8 finished with value: 3.6688859462738037 and parameters: {'lr': 0.006091664253368849, 'wd': 0.0005384461536853345}. Best is trial 7 with value: 3.631498098373413.
[I 2023-03-14 10:09:11,127] Trial 9 finished with value: 4.261521339416504 and parameters: {'lr': 0.0018141755070389311, 'wd': 0.0005129473767607587}. Best is trial 7 with value: 3.631498098373413.
{'lr': 0.008819657239070081, 'wd': 0.0009277374148896613}
3.631498098373413
"""
