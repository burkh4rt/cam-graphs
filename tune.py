#!/usr/bin/env python3

"""
Tunes hyperparameters with optuna
"""

import datetime
import os
import warnings

import matplotlib.pyplot as plt
import optuna as opt
import optuna.visualization.matplotlib as opt_mpl
import sklearn.metrics as skl_mets
import torch as t
import torch_geometric.loader as t_loader

import dataset
import model

warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

t.manual_seed(0)
criterion = t.nn.functional.binary_cross_entropy


def loss_val(mdl):
    mdl.eval()
    return criterion(
        mdl(
            dataset.batch_val.x,
            dataset.batch_val.edge_index,
            dataset.batch_val.edge_attr,
            dataset.batch_val.batch,
            dataset.batch_val.y[:, 1:],
        ),
        dataset.batch_val.y[:, 0].reshape(-1, 1),
    )


def loss_test(mdl):
    mdl.eval()
    return criterion(
        mdl(
            dataset.batch_test.x,
            dataset.batch_test.edge_index,
            dataset.batch_test.edge_attr,
            dataset.batch_test.batch,
            dataset.batch_test.y[:, 1:],
        ),
        dataset.batch_test.y[:, 0].reshape(-1, 1),
    )


def objective(trial):
    mdl = model.GCN(
        alpha_dropout=trial.suggest_float("alpha_dropout", 0.0, 1e-1),
        gat_heads=trial.suggest_int("gat_heads", 1, 3),
        gat_out_channels=trial.suggest_int("gat_out_channels", 1, 3),
        dim_penultimate=trial.suggest_int("dim_penultimate", 5, 25, step=5),
    )
    opt_type = trial.suggest_categorical("optimizer", ["Adagrad", "Adam"])
    optimizer = getattr(t.optim, opt_type)(
        mdl.parameters(),
        lr=trial.suggest_float("lr", 1e-3, 1e-2),
        weight_decay=trial.suggest_float("wd", 1e-4, 1e-3),
    )

    for epoch in range(trial.suggest_int("n_epochs", 5, 25, step=5)):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train,
                batch_size=trial.suggest_int(
                    "batch_size", 2**8, 2**10, log=True
                ),
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

        trial.report(loss_val(mdl), epoch)
        if trial.should_prune():
            raise opt.TrialPruned()

    t.save(
        mdl.state_dict(),
        os.path.join(
            os.path.dirname(__file__),
            "tmp",
            "{sn}-{tn}.ckpt".format(sn=study.study_name, tn=trial.number),
        ),
    )

    return loss_val(mdl)


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
        sampler=opt.samplers.TPESampler(
            multivariate=True, group=True, seed=42
        ),
        pruner=opt.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=3
        ),
    )
    study.optimize(objective, n_trials=25)

    print(study.best_params)
    print("val mse:  {:.3f}".format(study.best_value))

    mdl = model.GCN(
        **{
            k: v
            for k, v in study.best_params.items()
            if k
            in [
                "alpha_dropout",
                "gat_heads",
                "gat_out_channels",
                "dim_penultimate",
            ]
        }
    )
    mdl.load_state_dict(
        t.load(
            os.path.join(
                os.path.dirname(__file__),
                "tmp",
                "{sn}-{tn}.ckpt".format(
                    sn=study.study_name, tn=study.best_trial.number
                ),
            )
        )
    )
    mdl.eval()

    ypred = (
        mdl(
            dataset.batch_test.x,
            dataset.batch_test.edge_index,
            dataset.batch_test.edge_attr,
            dataset.batch_test.batch,
            dataset.batch_test.y[:, 1:],
        )
        .detach()
        .numpy()
        .ravel()
    )

    ytrue = dataset.batch_test.y[:, 0].numpy().ravel()

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

    opt_mpl.plot_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    plt.show()

    imps = opt.importance.get_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    feats_by_imps = sorted(imps.keys(), key=lambda k: imps[k], reverse=True)
    opt_mpl.plot_contour(study, params=feats_by_imps[:2])
    plt.show()

"""
[I 2023-04-04 09:56:06,385] A new study created in memory with name: gnns-graphs-tuning-20230404T0856Z
[I 2023-04-04 10:08:11,592] Trial 0 finished with value: 0.48120546340942383 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.0008795585311974417, 'n_epochs': 20, 'batch_size': 683}. Best is trial 0 with value: 0.48120546340942383.
[I 2023-04-04 10:16:59,559] Trial 1 finished with value: 0.3984091877937317 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 10:24:08,139] Trial 2 finished with value: 0.4141380786895752 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.0005628109945722505, 'n_epochs': 15, 'batch_size': 273}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 10:25:04,431] Trial 3 pruned.
[I 2023-04-04 10:26:06,439] Trial 4 pruned.
[I 2023-04-04 10:28:22,868] Trial 5 pruned.
[I 2023-04-04 10:29:10,508] Trial 6 finished with value: 0.41627711057662964 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.0008219772826786357, 'n_epochs': 5, 'batch_size': 1006}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 10:30:08,020] Trial 7 pruned.
[I 2023-04-04 10:32:06,821] Trial 8 pruned.
[I 2023-04-04 10:33:14,458] Trial 9 pruned.
[I 2023-04-04 10:35:13,154] Trial 10 pruned.
[I 2023-04-04 10:39:06,912] Trial 11 finished with value: 0.4120657444000244 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.00026791305299743223, 'n_epochs': 25, 'batch_size': 540}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 10:41:13,154] Trial 12 pruned.
[I 2023-04-04 10:41:21,121] Trial 13 pruned.
[I 2023-04-04 10:42:18,658] Trial 14 pruned.
[I 2023-04-04 10:42:27,127] Trial 15 pruned.
[I 2023-04-04 10:45:00,684] Trial 16 finished with value: 0.4126831293106079 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 2, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0002678666593598688, 'n_epochs': 5, 'batch_size': 580}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 10:46:01,225] Trial 17 pruned.
[I 2023-04-04 10:51:02,813] Trial 18 pruned.
[I 2023-04-04 10:52:09,709] Trial 19 pruned.
[I 2023-04-04 11:05:58,838] Trial 20 finished with value: 0.39883148670196533 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.0009073992339573194, 'n_epochs': 25, 'batch_size': 755}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 11:08:06,755] Trial 21 pruned.
[I 2023-04-04 11:17:27,207] Trial 22 finished with value: 0.4921698272228241 and parameters: {'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.0006971515921972502, 'n_epochs': 5, 'batch_size': 319}. Best is trial 1 with value: 0.3984091877937317.
[I 2023-04-04 11:18:34,443] Trial 23 pruned.
[I 2023-04-04 11:19:41,676] Trial 24 pruned.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}
val mse:  0.398
auc: 0.89
acc: 0.80
f1:  0.80
"""
