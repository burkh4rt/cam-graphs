#!/usr/bin/env python3

"""
Performs cross-validated graph-level regression;
tunes hyperparameters with optuna;
uses shap to explain feature importance
"""

import functools

import torch as t
import torch_geometric.loader as t_loader
import wandb as wb

import dataset as data
import model

t.manual_seed(0)
criterion = t.nn.L1Loss()

config_sweep = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "lv"},
    "parameters": {
        "alpha_dropout": {
            "distribution": "uniform",
            "min": 1e-4,
            "max": 1e-2,
        },
        "gat_heads": {"values": [1, 2, 3, 4, 5]},
        "gat_out_channels": {"values": [1, 2, 3, 4, 5]},
        "dim_penultimate": {"values": [5, 10, 15, 20, 25]},
        "optimizer": {"values": ["Adam", "Adagrad"]},
        "lr": {"distribution": "uniform", "min": 1e-4, "max": 1e-2},
        "wd": {"distribution": "uniform", "min": 1e-4, "max": 0.1},
        "n_epochs": {"values": [50, 60, 70, 80, 90, 100]},
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 7,
            "min": 2,
            "q": 2,
        },
    },
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 27,
    },
}

config_defaults = {
    "alpha_dropout": 1e-3,
    "gat_heads": 1,
    "gat_out_channels": 1,
    "dim_penultimate": 10,
    "optimizer": "Adam",
    "lr": 1e-3,
    "wd": 1e-3,
    "n_epochs": 70,
    "batch_size": 2**7,
}


def loss_val(mdl: t.nn.Module, dataset: data.dataset):
    """model `mdl` loss on validation set"""

    mdl.eval()
    return float(
        criterion(
            mdl(
                dataset.batch_val.x,
                dataset.batch_val.edge_index,
                dataset.batch_val.edge_attr,
                dataset.batch_val.batch,
                dataset.batch_val.y[:, 1:],
            ),
            dataset.batch_val.y[:, 0].reshape(-1, 1),
        ).detach()
    )


def loss_test(mdl: t.nn.Module, dataset: data.dataset):
    """model `mdl` loss on test set"""

    mdl.eval()
    return float(
        criterion(
            mdl(
                dataset.batch_test.x,
                dataset.batch_test.edge_index,
                dataset.batch_test.edge_attr,
                dataset.batch_test.batch,
                dataset.batch_test.y[:, 1:],
            ),
            dataset.batch_test.y[:, 0].reshape(-1, 1),
        ).detach()
    )


def loss_null(dataset: data.dataset):
    """loss from predicting the training mean on the test set"""
    return float(
        criterion(
            dataset.mean_train
            * t.ones_like(dataset.batch_test.y[:, 0].reshape(-1, 1)),
            dataset.batch_test.y[:, 0].reshape(-1, 1),
        ).detach()
    )


def objective(dataset: data.dataset):
    """helper function for hyperparameter tuning"""

    mdl = model.GCN(
        alpha_dropout=wb.config.alpha_dropout,
        gat_heads=wb.config.gat_heads,
        gat_out_channels=wb.config.gat_out_channels,
        dim_penultimate=wb.config.dim_penultimate,
        mean_train=dataset.mean_train,
        std_train=dataset.std_train,
    )
    optimizer = getattr(t.optim, wb.config.optimizer)(
        mdl.parameters(),
        lr=wb.config.lr,
        weight_decay=wb.config.wd,
    )

    for epoch in range(wb.config.n_epochs):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train,
                batch_size=wb.config.batch_size,
                shuffle=True,
                drop_last=True,  # otherwise we may pass a batch of size 1
            )
        ):
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
        lv = loss_val(mdl, dataset)
        lt = loss_test(mdl, dataset)
        print({"epoch": epoch, "lv": lv, "lt": lt})
        try:
            wb.log({"epoch": epoch, "lv": lv, "lt": lt})
        except BrokenPipeError:
            pass
    return lv


if __name__ == "__main__":

    wb.login()

    ds = data.dataset(0)

    run = wb.init(
        project="cam-graphs",
        config={"nested_param": {"manual_key": 1}} | config_defaults,
    )
    sweep_id = wb.sweep(config_sweep, project="cam-graphs")
    wb.agent(
        sweep_id, function=functools.partial(objective, dataset=ds), count=10
    )

    try:
        wb.finish()
    except BrokenPipeError:
        pass
