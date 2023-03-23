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
import pandas as pd
import shap
import torch as t
import torch_geometric.loader as t_loader

import dataset as data
import model

plt.rcParams["figure.constrained_layout.use"] = True
warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

t.manual_seed(0)
criterion = t.nn.MSELoss()


def model_fold(i_fold):
    dataset = data.dataset(i_fold)

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
            dim_penultimate=trial.suggest_int(
                "dim_penultimate", 5, 25, step=5
            ),
            mean_train=dataset.mean_train,
            std_train=dataset.std_train,
        )
        opt_type = trial.suggest_categorical("optimizer", ["Adagrad", "Adam"])
        optimizer = getattr(t.optim, opt_type)(
            mdl.parameters(),
            lr=trial.suggest_float("lr", 1e-3, 1e-2),
            weight_decay=trial.suggest_float("wd", 1e-4, 1e-2),
        )

        for epoch in range(trial.suggest_int("n_epochs", 50, 100, step=10)):
            mdl.train()
            for data in iter(
                t_loader.DataLoader(
                    dataset.data_train,
                    batch_size=2 ** trial.suggest_int("batch_log2", 2, 4),
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

    study = opt.create_study(
        direction="minimize",
        study_name="-".join(
            [
                os.path.basename(os.path.dirname(__file__)),
                "tuning",
                f"{i_fold=}",
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
        | {"mean_train": dataset.mean_train, "std_train": dataset.std_train}
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

    # plt.subplots()
    # opt_mpl.plot_param_importances(
    #     study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    # )
    # plt.savefig(os.path.join("figures", f"optuna_imp_{study.study_name}.pdf"))
    # plt.show()

    # plt.subplots()
    # imps = opt.importance.get_param_importances(
    #     study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    # )
    # feats_by_imps = sorted(imps.keys(), key=lambda k: imps[k], reverse=True)
    # opt_mpl.plot_contour(study, params=feats_by_imps[:2])
    # plt.savefig(
    #     os.path.join("figures", f"optuna_contour_{study.study_name}.pdf")
    # )
    # plt.show()

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
    # plt.subplots()
    # shap.plots.bar(shap_values, show=False)
    # plt.savefig(os.path.join("figures", f"shap_{study.study_name}.pdf"))
    #
    # plt.subplots()
    # shap.summary_plot(
    #     shap_values,
    #     array_test,
    #     plot_type="layered_violin",
    #     color="coolwarm",
    #     show=False,
    # )
    # plt.savefig(os.path.join("figures", f"shap_violin_{study.study_name}.pdf"))

    preds = (
        mdl(
            dataset.batch_test.x,
            dataset.batch_test.edge_index,
            dataset.batch_test.edge_attr,
            dataset.batch_test.batch,
            dataset.batch_test.y[:, 1:],
        )
        .detach()
        .numpy()
    )
    ages = dataset.batch_test.y[:, 0].detach().numpy()
    ids = dataset.test_ids
    return pd.DataFrame(
        index=ids, data={"ages": ages.ravel(), "preds": preds.ravel()}
    )


if __name__ == "__main__":
    results = list()
    for i in range(data.n_folds):
        results.append(model_fold(i))
    pd.concat(results, axis=0).to_csv(
        os.path.join("data", "age_predictions.csv")
    )

"""
[I 2023-03-22 16:48:34,605] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230322T1648Z
[I 2023-03-22 16:49:27,027] Trial 0 finished with value: 11.035832405090332 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 11.035832405090332.
[I 2023-03-22 16:50:23,455] Trial 1 finished with value: 12.510648727416992 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 11.035832405090332.
[I 2023-03-22 16:50:24,576] Trial 2 pruned.
[I 2023-03-22 16:50:25,597] Trial 3 pruned.
[I 2023-03-22 16:50:31,589] Trial 4 pruned.
{'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}
val mse:  11.036
test mse: 15.027
null mse: 45.881
[I 2023-03-22 16:50:31,917] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230322T1650Z
[I 2023-03-22 16:51:22,816] Trial 0 finished with value: 14.802358627319336 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.802358627319336.
[I 2023-03-22 16:52:15,618] Trial 1 finished with value: 12.924217224121094 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 12.924217224121094.
[I 2023-03-22 16:52:57,513] Trial 2 finished with value: 14.011611938476562 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 12.924217224121094.
[I 2023-03-22 16:52:58,472] Trial 3 pruned.
[I 2023-03-22 16:54:21,268] Trial 4 finished with value: 17.370635986328125 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 12.924217224121094.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}
val mse:  12.924
test mse: 10.694
null mse: 30.226
[I 2023-03-22 16:54:21,629] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230322T1654Z
[I 2023-03-22 16:57:12,867] Trial 0 finished with value: 16.589147567749023 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 16.589147567749023.
[I 2023-03-22 17:02:02,229] Trial 1 finished with value: 14.209061622619629 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 14.209061622619629.
[I 2023-03-22 17:36:05,371] Trial 2 finished with value: 13.928359985351562 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 13.928359985351562.
[I 2023-03-22 17:36:43,774] Trial 3 finished with value: 13.062106132507324 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 13.062106132507324.
[I 2023-03-22 17:36:45,797] Trial 4 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  13.062
test mse: 14.236
null mse: 35.465
[I 2023-03-22 17:36:46,072] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230322T1736Z
[I 2023-03-22 17:37:31,204] Trial 0 finished with value: 23.890073776245117 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 23.890073776245117.
[I 2023-03-22 17:38:20,395] Trial 1 finished with value: 19.88289451599121 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 19.88289451599121.
[I 2023-03-22 17:38:21,387] Trial 2 pruned.
[I 2023-03-22 17:39:00,878] Trial 3 finished with value: 18.8381404876709 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 18.8381404876709.
[I 2023-03-22 17:39:01,942] Trial 4 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  18.838
test mse: 13.775
null mse: 31.958
"""
