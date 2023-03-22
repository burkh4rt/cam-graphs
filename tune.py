#!/usr/bin/env python3

"""
Tunes hyperparameters with optuna
"""

import datetime
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna as opt
import optuna.visualization.matplotlib as opt_mpl
import shap
import torch as t
import torch_geometric.loader as t_loader

import dataset
import model

plt.rcParams["figure.constrained_layout.use"] = True
warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

t.manual_seed(0)
criterion = t.nn.MSELoss()


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


def loss_null():
    return criterion(
        dataset.mean_train
        * t.ones_like(dataset.batch_test.y[:, 0].reshape(-1, 1)),
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
        weight_decay=trial.suggest_float("wd", 1e-5, 1e-4),
    )

    for epoch in range(trial.suggest_int("n_epochs", 5, 10, step=5)):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train,
                batch_size=1000,
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

    print("test mse: {:.3f}".format(loss_test(mdl).detach().numpy()))
    print("null mse: {:.3f}".format(loss_null().detach().numpy()))

    plt.subplots()
    opt_mpl.plot_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    plt.savefig(os.path.join("figures", f"optuna_imp_{study.study_name}.pdf"))
    plt.show()

    plt.subplots()
    imps = opt.importance.get_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    feats_by_imps = sorted(imps.keys(), key=lambda k: imps[k], reverse=True)
    opt_mpl.plot_contour(study, params=feats_by_imps[:2])
    plt.savefig(
        os.path.join("figures", f"optuna_contour_{study.study_name}.pdf")
    )
    plt.show()

    # array_test = np.row_stack(
    #     [
    #         np.concatenate(
    #             [
    #                 d.x.detach().numpy().reshape(-1),
    #                 d.y[:, 1:].detach().numpy().reshape(-1),
    #             ]
    #         )
    #         for d in dataset.data_test
    #     ]
    # )

    # explainer = shap.Explainer(
    #     mdl.as_function_of_x_y,
    #     array_test,
    #     feature_names=list(dataset.cols_xy1_ravelled),
    # )
    # shap_values = explainer(array_test)
    #
    # fig, ax = plt.subplots()
    # shap.plots.bar(shap_values, show=False)
    # plt.savefig(os.path.join("figures", f"shap_{study.study_name}.pdf"))

"""
[I 2023-03-22 11:47:49,013] A new study created in memory with name: gnns-graphs-tuning-20230322T1147Z
[I 2023-03-22 11:50:41,216] Trial 0 finished with value: 3.808786630630493 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 8.795585311974417e-05, 'n_epochs': 10}. Best is trial 0 with value: 3.808786630630493.
[I 2023-03-22 11:52:46,032] Trial 1 finished with value: 3.553706169128418 and parameters: {'alpha_dropout': 0.07080725777960455, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0026506405886809045, 'wd': 3.7381801866358395e-05, 'n_epochs': 10}. Best is trial 1 with value: 3.553706169128418.
[I 2023-03-22 11:53:12,792] Trial 2 pruned.
[I 2023-03-22 11:55:27,653] Trial 3 finished with value: 3.5279109477996826 and parameters: {'alpha_dropout': 0.05142344384136116, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 20, 'optimizer': 'Adagrad', 'lr': 0.00953996983528, 'wd': 9.690688297671034e-05, 'n_epochs': 10}. Best is trial 3 with value: 3.5279109477996826.
[I 2023-03-22 11:56:29,409] Trial 4 finished with value: 3.7166733741760254 and parameters: {'alpha_dropout': 0.03046137691733707, 'gat_heads': 1, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0013094966900369656, 'wd': 9.183883618709039e-05, 'n_epochs': 5}. Best is trial 3 with value: 3.5279109477996826.
{'alpha_dropout': 0.05142344384136116, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 20, 'optimizer': 'Adagrad', 'lr': 0.00953996983528, 'wd': 9.690688297671034e-05, 'n_epochs': 10}
val mse:  3.528
test mse: 3.437
null mse: 3.978
"""
