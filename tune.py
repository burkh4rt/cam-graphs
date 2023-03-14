#!/usr/bin/env python3

"""
Tunes hyperparameters with optuna
"""

import datetime
import os
import warnings

import optuna as opt
import torch as t
import torch_geometric.loader as t_loader

import dataset
import model

warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

t.manual_seed(0)
n_epochs = 10

batch_val = next(
    iter(
        t_loader.DataLoader(
            dataset.data_val,
            batch_size=len(dataset.val_ids),
            shuffle=False,
        )
    )
)


def loss_val(mdl, criterion):
    mdl.eval()
    return criterion(
        mdl(
            batch_val.x,
            batch_val.edge_index,
            batch_val.edge_attr,
            batch_val.batch,
            batch_val.y[:, 1:],
        ),
        batch_val.y[:, 0].reshape(-1, 1),
    )


def objective(trial):
    mdl = model.GCN(
        alpha_dropout=trial.suggest_float("alpha_dropout", 0.0, 1e-1),
        gat_heads=trial.suggest_int("gat_heads", 1, 5),
        gat_out_channels=trial.suggest_int("gat_out_channels", 1, 5),
        dim_penultimate=trial.suggest_int("dim_penultimate", 5, 25, step=5),
    )
    opt_type = trial.suggest_categorical("optimizer", ["Adagrad", "Adam"])
    optimizer = getattr(t.optim, opt_type)(
        mdl.parameters(),
        lr=trial.suggest_float("lr", 1e-3, 1e-2),
        weight_decay=trial.suggest_float("wd", 1e-4, 1e-3),
    )
    criterion = t.nn.MSELoss()

    for epoch in range(n_epochs):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train, batch_size=1000, shuffle=True
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

        trial.report(loss_val(mdl, criterion), epoch)
        if trial.should_prune():
            raise opt.TrialPruned()

    return loss_val(mdl, criterion)


if __name__ == "__main__":
    study = opt.create_study(
        direction="minimize",
        study_name="-".join(
            [
                os.path.basename(os.path.dirname(__file__)),
                "tuning",
                datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y%m%dT%H%MZ"
                ),
            ]
        ),
        sampler=opt.samplers.TPESampler(multivariate=True, group=True),
        pruner=opt.pruners.HyperbandPruner(
            min_resource=1, max_resource=n_epochs, reduction_factor=3
        ),
    )
    study.optimize(objective, n_trials=30)

    print(study.best_params)
    print(study.best_value)

"""
[I 2023-03-14 11:54:19,587] A new study created in memory with name: ukbb-graphs-tuning-20230314T1154Z
[I 2023-03-14 11:55:40,891] Trial 0 finished with value: 3.8084070682525635 and parameters: {'alpha_dropout': 0.08105967709130209, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.004860338271132598, 'wd': 0.00012238850127290908}. Best is trial 0 with value: 3.8084070682525635.
[I 2023-03-14 11:57:44,204] Trial 1 finished with value: 3.539700746536255 and parameters: {'alpha_dropout': 0.011248063481413318, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00816790214807728, 'wd': 0.00040995633487050454}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:07:44,346] Trial 2 finished with value: 3.657808542251587 and parameters: {'alpha_dropout': 0.09286390844379899, 'gat_heads': 3, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.008516955738167895, 'wd': 0.00039512994888576047}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:09:32,707] Trial 3 finished with value: 3.693530559539795 and parameters: {'alpha_dropout': 0.07916029000991343, 'gat_heads': 3, 'gat_out_channels': 4, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.004798612090518737, 'wd': 0.0006314490042947818}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:10:55,824] Trial 4 finished with value: 3.5682523250579834 and parameters: {'alpha_dropout': 0.02794630488094271, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.007743051502490038, 'wd': 0.00013877923918828244}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:11:19,950] Trial 5 pruned.
[I 2023-03-14 12:12:57,635] Trial 6 finished with value: 3.5425984859466553 and parameters: {'alpha_dropout': 0.014419933111480078, 'gat_heads': 4, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.009934788270997644, 'wd': 0.00030922777785115065}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:13:20,885] Trial 7 pruned.
[I 2023-03-14 12:14:37,225] Trial 8 finished with value: 3.5732171535491943 and parameters: {'alpha_dropout': 0.030081956861242556, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0064071839732298555, 'wd': 0.0001960966427014635}. Best is trial 1 with value: 3.539700746536255.
[I 2023-03-14 12:14:52,764] Trial 9 pruned.
[I 2023-03-14 12:15:19,130] Trial 10 pruned.
[I 2023-03-14 12:15:53,764] Trial 11 pruned.
[I 2023-03-14 12:16:13,557] Trial 12 pruned.
[I 2023-03-14 12:17:18,314] Trial 13 pruned.
[I 2023-03-14 12:18:00,834] Trial 14 pruned.
[I 2023-03-14 12:18:17,177] Trial 15 pruned.
[I 2023-03-14 12:18:59,701] Trial 16 pruned.
[I 2023-03-14 12:20:19,734] Trial 17 finished with value: 3.5391483306884766 and parameters: {'alpha_dropout': 0.02242681321024366, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.007078301509604504, 'wd': 0.00015608655671708825}. Best is trial 17 with value: 3.5391483306884766.
[I 2023-03-14 12:20:43,488] Trial 18 pruned.
[I 2023-03-14 12:21:33,276] Trial 19 pruned.
[I 2023-03-14 12:23:19,779] Trial 20 finished with value: 3.530109405517578 and parameters: {'alpha_dropout': 0.0011581007179390479, 'gat_heads': 5, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.006268860239378637, 'wd': 0.0009620241760413245}. Best is trial 20 with value: 3.530109405517578.
[I 2023-03-14 12:24:09,682] Trial 21 pruned.
[I 2023-03-14 12:24:50,668] Trial 22 pruned.
[I 2023-03-14 12:25:29,788] Trial 23 pruned.
[I 2023-03-14 12:26:48,937] Trial 24 pruned.
[I 2023-03-14 12:27:35,710] Trial 25 pruned.
[I 2023-03-14 12:29:12,132] Trial 26 pruned.
[I 2023-03-14 12:30:43,686] Trial 27 finished with value: 3.483602285385132 and parameters: {'alpha_dropout': 0.003148939363068679, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.009131823568546747, 'wd': 0.0002085365192668435}. Best is trial 27 with value: 3.483602285385132.
[I 2023-03-14 12:32:31,030] Trial 28 finished with value: 3.540365695953369 and parameters: {'alpha_dropout': 0.0014907933174646893, 'gat_heads': 3, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002790039393731718, 'wd': 0.0006586332832035749}. Best is trial 27 with value: 3.483602285385132.
[I 2023-03-14 12:33:07,566] Trial 29 pruned.
{'alpha_dropout': 0.003148939363068679, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.009131823568546747, 'wd': 0.0002085365192668435}
3.483602285385132
"""
