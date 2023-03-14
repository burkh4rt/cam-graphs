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
    mdl = model.GCN()
    optimizer = t.optim.Adagrad(
        mdl.parameters(),
        lr=trial.suggest_float("lr", 1e-3, 1e-2),
        weight_decay=trial.suggest_float("wd", 1e-4, 1e-3),
    )
    criterion = t.nn.MSELoss()

    for epoch in range(n_epochs):
        mdl.train()
        loader_train = t_loader.DataLoader(
            dataset.data_train, batch_size=1000, shuffle=True
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
[I 2023-03-14 10:48:59,978] A new study created in memory with name: ukbb-graphs-tuning-20230314T1048Z
[I 2023-03-14 10:51:29,419] Trial 0 finished with value: 3.624445676803589 and parameters: {'lr': 0.006187822405193944, 'wd': 0.0009578319827298369}. Best is trial 0 with value: 3.624445676803589.
[I 2023-03-14 10:53:55,448] Trial 1 finished with value: 3.8627800941467285 and parameters: {'lr': 0.004639688241174073, 'wd': 0.0002567076530790423}. Best is trial 0 with value: 3.624445676803589.
[I 2023-03-14 10:54:53,066] Trial 2 pruned.
[I 2023-03-14 10:57:17,346] Trial 3 finished with value: 3.6381077766418457 and parameters: {'lr': 0.009738553844390353, 'wd': 0.00035438660215031886}. Best is trial 0 with value: 3.624445676803589.
[I 2023-03-14 10:58:15,541] Trial 4 pruned.
[I 2023-03-14 10:58:44,920] Trial 5 pruned.
[I 2023-03-14 11:01:11,717] Trial 6 finished with value: 3.789339780807495 and parameters: {'lr': 0.004764868998512587, 'wd': 0.000724102559394082}. Best is trial 0 with value: 3.624445676803589.
[I 2023-03-14 11:01:40,940] Trial 7 pruned.
[I 2023-03-14 11:02:10,282] Trial 8 pruned.
[I 2023-03-14 11:02:39,637] Trial 9 pruned.
[I 2023-03-14 11:03:49,569] Trial 10 pruned.
[I 2023-03-14 11:06:23,304] Trial 11 pruned.
[I 2023-03-14 11:07:23,941] Trial 12 pruned.
[I 2023-03-14 11:07:54,559] Trial 13 pruned.
[I 2023-03-14 11:08:54,686] Trial 14 pruned.
[I 2023-03-14 11:11:26,383] Trial 15 pruned.
[I 2023-03-14 11:12:25,759] Trial 16 pruned.
[I 2023-03-14 11:12:56,183] Trial 17 pruned.
[I 2023-03-14 11:13:58,005] Trial 18 pruned.
[I 2023-03-14 11:14:58,761] Trial 19 pruned.
[I 2023-03-14 11:15:59,201] Trial 20 pruned.
[I 2023-03-14 11:18:28,944] Trial 21 pruned.
[I 2023-03-14 11:19:28,872] Trial 22 pruned.
[I 2023-03-14 11:20:27,929] Trial 23 pruned.
[I 2023-03-14 11:22:57,626] Trial 24 finished with value: 3.6197071075439453 and parameters: {'lr': 0.009560898430121804, 'wd': 0.0002490963395301151}. Best is trial 24 with value: 3.6197071075439453.
[I 2023-03-14 11:23:27,899] Trial 25 pruned.
[I 2023-03-14 11:24:27,428] Trial 26 pruned.
[I 2023-03-14 11:25:26,710] Trial 27 pruned.
[I 2023-03-14 11:27:55,412] Trial 28 finished with value: 3.625743865966797 and parameters: {'lr': 0.009553121819816402, 'wd': 0.00010210813146089933}. Best is trial 24 with value: 3.6197071075439453.
[I 2023-03-14 11:28:55,047] Trial 29 pruned.
{'lr': 0.009560898430121804, 'wd': 0.0002490963395301151}
3.6197071075439453
"""
