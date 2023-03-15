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

    for epoch in range(trial.suggest_int("n_epochs", 5, 25, step=5)):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train,
                batch_size=trial.suggest_int(
                    "batch_size", 2**8, 2**10, log=True
                ),
                shuffle=True,
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
    study.optimize(objective, n_trials=5)

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

    ypred = mdl(
        dataset.batch_test.x,
        dataset.batch_test.edge_index,
        dataset.batch_test.edge_attr,
        dataset.batch_test.batch,
        dataset.batch_test.y[:, 1:],
    ).detach().numpy().ravel()

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
[I 2023-03-15 09:30:51,379] A new study created in memory with name: ukbb-graphs-tuning-20230315T0930Z
[I 2023-03-15 09:37:10,479] Trial 0 finished with value: 0.5443597435951233 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.0008795585311974417, 'n_epochs': 20, 'batch_size': 683}. Best is trial 0 with value: 0.5443597435951233.
[I 2023-03-15 09:41:52,236] Trial 1 finished with value: 0.40592294931411743 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}. Best is trial 1 with value: 0.40592294931411743.
[I 2023-03-15 09:42:16,460] Trial 2 pruned.
[I 2023-03-15 09:46:29,541] Trial 3 finished with value: 0.45701220631599426 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.00018790490260574548, 'n_epochs': 20, 'batch_size': 471}. Best is trial 1 with value: 0.40592294931411743.
[I 2023-03-15 09:47:23,661] Trial 4 pruned.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}
val mse:  0.406
auc: 0.89
acc: 0.80
f1:  0.81
"""
