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
import torch_geometric as pyg
import torch_geometric.loader as t_loader

import dataset
import model

warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

pyg.seed_everything(0)
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

    array_test = np.row_stack(
        [
            np.concatenate(
                [
                    d.x.detach().numpy().reshape(-1),
                    d.edge_attr.detach().numpy().reshape(-1),
                    d.y[:, 1:].detach().numpy().reshape(-1),
                ]
            )
            for d in dataset.data_test
        ]
    )

    # too many variables for shap -- we get exit code 137 SIGKILL
    # explainer = shap.explainers.Permutation(
    #     mdl.as_function_of_x_attr_y,
    #     array_test,
    #     feature_names=list(dataset.names_x_attr_y_ravelled),
    #     seed=0,
    #     max_evals=7702
    # )
    # shap_values = explainer(array_test)
    #
    # fig = plt.figure()
    # shap.plots.bar(shap_values, show=False)
    # fig.savefig(os.path.join("figures", f"shap_{study.study_name}.pdf"))
    # plt.gcf().show()

"""
[I 2023-04-03 14:51:48,629] A new study created in memory with name: gnns-graphs-tuning-20230403T1351Z
[I 2023-04-03 15:03:13,808] Trial 0 finished with value: 33.98548126220703 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.0008795585311974417, 'n_epochs': 20, 'batch_size': 683}. Best is trial 0 with value: 33.98548126220703.
[I 2023-04-03 15:11:54,338] Trial 1 finished with value: 28.95796775817871 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}. Best is trial 1 with value: 28.95796775817871.
[I 2023-04-03 15:19:08,308] Trial 2 finished with value: 30.795333862304688 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.0005628109945722505, 'n_epochs': 15, 'batch_size': 273}. Best is trial 1 with value: 28.95796775817871.
[I 2023-04-03 15:28:43,175] Trial 3 finished with value: 33.78669357299805 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.00018790490260574548, 'n_epochs': 20, 'batch_size': 471}. Best is trial 1 with value: 28.95796775817871.
[I 2023-04-03 15:36:21,101] Trial 4 finished with value: 29.377939224243164 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.0005680612190600297, 'n_epochs': 15, 'batch_size': 330}. Best is trial 1 with value: 28.95796775817871.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.000572280788469014, 'n_epochs': 15, 'batch_size': 383}
val mse:  28.958
test mse: 31.001
null mse: 51.434
"""
