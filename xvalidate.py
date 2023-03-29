#!/usr/bin/env python3

"""
Performs cross-validated graph-level regression;
tunes hyperparameters with optuna;
uses shap to explain feature importance
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


def model_fold(
    i_fold: int,
    *,
    explain_features: bool = False,
    explain_training: bool = False,
):
    """train and test on fold `i_fold` of the dataset"""

    dataset = data.dataset(i_fold)

    def loss_val(mdl: t.nn.Module):
        """model `mdl` loss on validation set"""

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

    def loss_test(mdl: t.nn.Module):
        """model `mdl` loss on test set"""

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
        """loss from predicting the training mean on the test set"""
        return criterion(
            dataset.mean_train
            * t.ones_like(dataset.batch_test.y[:, 0].reshape(-1, 1)),
            dataset.batch_test.y[:, 0].reshape(-1, 1),
        )

    def objective(trial):
        """helper function for hyperparameter tuning"""

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
                    batch_size=2 ** trial.suggest_int("batch_log2", 2, 6),
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

    # do the optimising
    study.optimize(objective, n_trials=25)

    # report results from optimisation
    print(study.best_params)
    print("val mse:  {:.3f}".format(study.best_value))

    # load best-performing model
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

    # report performance on test set from best validated model
    print("test mse: {:.3f}".format(loss_test(mdl).detach().numpy()))
    print("null mse: {:.3f}".format(loss_null().detach().numpy()))

    if explain_training:
        opt_mpl.plot_param_importances(
            study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
        )
        plt.gcf().savefig(
            os.path.join("figures", f"optuna_imp_{study.study_name}.pdf")
        )
        plt.gcf().show()

        imps = opt.importance.get_param_importances(
            study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
        )
        feats_by_imps = sorted(
            imps.keys(), key=lambda k: imps[k], reverse=True
        )
        opt_mpl.plot_contour(study, params=feats_by_imps[:2])
        plt.gcf().savefig(
            os.path.join("figures", f"optuna_contour_{study.study_name}.pdf")
        )
        plt.gcf().show()

    if explain_features:
        array_test = np.row_stack(
            [
                np.concatenate(
                    [
                        d.x.detach().numpy().reshape(-1),
                        d.y[:, 1:].detach().numpy().reshape(-1),
                    ]
                )
                for d in dataset.data_test
            ]
        )

        explainer = shap.explainers.Permutation(
            mdl.as_function_of_x_y,
            array_test,
            feature_names=list(dataset.cols_xy1_ravelled),
            seed=0,
        )
        shap_values = explainer(array_test)

        fig = plt.figure()
        shap.plots.bar(shap_values, show=False)
        fig.savefig(os.path.join("figures", f"shap_{study.study_name}.pdf"))
        plt.gcf().show()

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
        index=ids,
        data={
            "ages": ages.ravel(),
            "preds": preds.ravel(),
            "null_preds": dataset.mean_train,
        },
    )


if __name__ == "__main__":
    # collect results from predicting on the test set of each fold
    results = pd.concat(
        [
            model_fold(
                i
            )  # , explain_features=(i == 0), explain_training=(i == 0))
            for i in range(data.n_folds)
        ],
        axis=0,
    )

    print("-" * 79)
    print(
        "overall mmse:  {:.2f}".format(
            np.mean(
                np.square(
                    results.ages.to_numpy().ravel()
                    - results.preds.to_numpy().ravel()
                )
            )
        )
    )
    print(
        "null mmse:  {:.2f}".format(
            np.mean(
                np.square(
                    results.ages.to_numpy().ravel()
                    - results.null_preds.to_numpy().ravel()
                )
            )
        )
    )
    print(
        "baseline mmse: {:.2f}".format(
            np.mean(
                np.square(
                    results.ages.to_numpy().ravel()
                    - results.ages.to_numpy().mean()
                )
            )
        )
    )

    results.to_csv(
        os.path.join(
            "data",
            "bacs_age_predictions_{}.csv".format(
                datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            ),
        )
    )

"""
[I 2023-03-29 12:42:57,385] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230329T1142Z
[I 2023-03-29 12:43:52,528] Trial 0 finished with value: 13.277200698852539 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 13.277200698852539.
[I 2023-03-29 12:44:56,487] Trial 1 finished with value: 10.110960006713867 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:44:57,987] Trial 2 pruned.
[I 2023-03-29 12:45:50,660] Trial 3 finished with value: 11.805442810058594 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:46:50,681] Trial 4 finished with value: 17.079105377197266 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:47:33,386] Trial 5 finished with value: 10.111681938171387 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:48:01,718] Trial 6 finished with value: 11.6946382522583 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 6}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:48:03,121] Trial 7 pruned.
[I 2023-03-29 12:49:15,038] Trial 8 finished with value: 22.02147102355957 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 1 with value: 10.110960006713867.
[I 2023-03-29 12:49:58,829] Trial 9 finished with value: 7.56199836730957 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 9 with value: 7.56199836730957.
[I 2023-03-29 12:50:00,575] Trial 10 pruned.
[I 2023-03-29 12:50:01,818] Trial 11 pruned.
[I 2023-03-29 12:50:04,842] Trial 12 pruned.
[I 2023-03-29 12:50:41,664] Trial 13 finished with value: 11.551963806152344 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}. Best is trial 9 with value: 7.56199836730957.
[I 2023-03-29 12:50:48,977] Trial 14 pruned.
[I 2023-03-29 12:50:50,078] Trial 15 pruned.
[I 2023-03-29 12:50:51,489] Trial 16 pruned.
[I 2023-03-29 12:50:52,940] Trial 17 pruned.
[I 2023-03-29 12:51:47,608] Trial 18 finished with value: 7.362910747528076 and parameters: {'alpha_dropout': 0.03410663510502585, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008354999801810942, 'wd': 0.005596488034834678, 'n_epochs': 80, 'batch_log2': 3}. Best is trial 18 with value: 7.362910747528076.
[I 2023-03-29 12:51:54,839] Trial 19 pruned.
[I 2023-03-29 12:52:01,756] Trial 20 pruned.
[I 2023-03-29 12:52:08,730] Trial 21 pruned.
[I 2023-03-29 12:52:56,733] Trial 22 finished with value: 7.503855228424072 and parameters: {'alpha_dropout': 0.05487337893665861, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.003928597283433409, 'wd': 0.00749026491066844, 'n_epochs': 80, 'batch_log2': 6}. Best is trial 18 with value: 7.362910747528076.
[I 2023-03-29 12:53:13,595] Trial 23 pruned.
[I 2023-03-29 12:53:15,867] Trial 24 pruned.
{'alpha_dropout': 0.03410663510502585, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008354999801810942, 'wd': 0.005596488034834678, 'n_epochs': 80, 'batch_log2': 3}
val mse:  7.363
test mse: 12.545
null mse: 20.241
[I 2023-03-29 12:53:16,145] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230329T1153Z
[I 2023-03-29 12:54:26,967] Trial 0 finished with value: 26.655885696411133 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 26.655885696411133.
[I 2023-03-29 12:55:34,586] Trial 1 finished with value: 29.499549865722656 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 26.655885696411133.
[I 2023-03-29 12:55:36,096] Trial 2 pruned.
[I 2023-03-29 12:55:37,410] Trial 3 pruned.
[I 2023-03-29 12:56:40,989] Trial 4 finished with value: 23.201560974121094 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 23.201560974121094.
[I 2023-03-29 12:56:42,842] Trial 5 pruned.
[I 2023-03-29 12:56:48,578] Trial 6 pruned.
[I 2023-03-29 12:57:38,808] Trial 7 finished with value: 24.78948974609375 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 4 with value: 23.201560974121094.
[I 2023-03-29 12:57:42,015] Trial 8 pruned.
[I 2023-03-29 12:57:43,791] Trial 9 pruned.
[I 2023-03-29 12:57:47,407] Trial 10 pruned.
[I 2023-03-29 12:58:56,584] Trial 11 finished with value: 25.893238067626953 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.001947043582971755, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 4 with value: 23.201560974121094.
[I 2023-03-29 12:58:58,262] Trial 12 pruned.
[I 2023-03-29 12:59:35,001] Trial 13 finished with value: 24.647632598876953 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}. Best is trial 4 with value: 23.201560974121094.
[I 2023-03-29 12:59:38,036] Trial 14 pruned.
[I 2023-03-29 12:59:39,353] Trial 15 pruned.
[I 2023-03-29 13:00:18,350] Trial 16 finished with value: 21.894582748413086 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 16 with value: 21.894582748413086.
[I 2023-03-29 13:00:19,364] Trial 17 pruned.
[I 2023-03-29 13:01:33,896] Trial 18 finished with value: 18.846330642700195 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 18 with value: 18.846330642700195.
[I 2023-03-29 13:01:36,804] Trial 19 pruned.
[I 2023-03-29 13:02:01,908] Trial 20 pruned.
[I 2023-03-29 13:02:04,803] Trial 21 pruned.
[I 2023-03-29 13:02:07,415] Trial 22 pruned.
[I 2023-03-29 13:02:13,534] Trial 23 pruned.
[I 2023-03-29 13:02:15,109] Trial 24 pruned.
{'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}
val mse:  18.846
test mse: 6.832
null mse: 18.394
[I 2023-03-29 13:02:15,386] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230329T1202Z
[I 2023-03-29 13:03:13,194] Trial 0 finished with value: 42.12550354003906 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 42.12550354003906.
[I 2023-03-29 13:04:14,063] Trial 1 finished with value: 37.68899154663086 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 1 with value: 37.68899154663086.
[I 2023-03-29 13:04:20,801] Trial 2 pruned.
[I 2023-03-29 13:04:21,986] Trial 3 pruned.
[I 2023-03-29 13:05:19,393] Trial 4 finished with value: 32.326908111572266 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 32.326908111572266.
[I 2023-03-29 13:05:21,045] Trial 5 pruned.
[I 2023-03-29 13:05:22,207] Trial 6 pruned.
[I 2023-03-29 13:05:40,637] Trial 7 pruned.
[I 2023-03-29 13:05:43,522] Trial 8 pruned.
[I 2023-03-29 13:05:46,733] Trial 9 pruned.
[I 2023-03-29 13:06:36,737] Trial 10 finished with value: 30.214378356933594 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 30.214378356933594.
[I 2023-03-29 13:06:39,279] Trial 11 pruned.
[I 2023-03-29 13:06:40,782] Trial 12 pruned.
[I 2023-03-29 13:06:55,539] Trial 13 pruned.
[I 2023-03-29 13:07:29,451] Trial 14 finished with value: 36.63700866699219 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 10 with value: 30.214378356933594.
[I 2023-03-29 13:07:44,268] Trial 15 pruned.
[I 2023-03-29 13:07:45,684] Trial 16 pruned.
[I 2023-03-29 13:07:48,394] Trial 17 pruned.
[I 2023-03-29 13:07:51,014] Trial 18 pruned.
[I 2023-03-29 13:07:52,437] Trial 19 pruned.
[I 2023-03-29 13:08:52,517] Trial 20 finished with value: 39.69758605957031 and parameters: {'alpha_dropout': 0.07903471339385376, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008659944129881898, 'wd': 0.002391868534318195, 'n_epochs': 90, 'batch_log2': 2}. Best is trial 10 with value: 30.214378356933594.
[I 2023-03-29 13:08:55,126] Trial 21 pruned.
[I 2023-03-29 13:09:15,470] Trial 22 pruned.
[I 2023-03-29 13:09:16,930] Trial 23 pruned.
[I 2023-03-29 13:09:19,545] Trial 24 pruned.
{'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}
val mse:  30.214
test mse: 17.686
null mse: 34.337
[I 2023-03-29 13:09:19,840] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230329T1209Z
[I 2023-03-29 13:10:15,905] Trial 0 finished with value: 19.14979362487793 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 19.14979362487793.
[I 2023-03-29 13:11:14,189] Trial 1 finished with value: 17.61363410949707 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 1 with value: 17.61363410949707.
[I 2023-03-29 13:12:04,412] Trial 2 finished with value: 17.598657608032227 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 17.598657608032227.
[I 2023-03-29 13:12:51,351] Trial 3 finished with value: 13.512393951416016 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 3 with value: 13.512393951416016.
[I 2023-03-29 13:12:58,055] Trial 4 pruned.
[I 2023-03-29 13:12:59,605] Trial 5 pruned.
[I 2023-03-29 13:13:01,922] Trial 6 pruned.
[I 2023-03-29 13:13:03,157] Trial 7 pruned.
[I 2023-03-29 13:13:05,718] Trial 8 pruned.
[I 2023-03-29 13:13:43,559] Trial 9 finished with value: 13.715553283691406 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 13.512393951416016.
[I 2023-03-29 13:13:45,130] Trial 10 pruned.
[I 2023-03-29 13:14:40,766] Trial 11 finished with value: 17.647199630737305 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.001947043582971755, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 3 with value: 13.512393951416016.
[I 2023-03-29 13:14:43,469] Trial 12 pruned.
[I 2023-03-29 13:14:44,530] Trial 13 pruned.
[I 2023-03-29 13:14:45,940] Trial 14 pruned.
[I 2023-03-29 13:14:48,069] Trial 15 pruned.
[I 2023-03-29 13:14:49,348] Trial 16 pruned.
[I 2023-03-29 13:14:51,879] Trial 17 pruned.
[I 2023-03-29 13:14:53,007] Trial 18 pruned.
[I 2023-03-29 13:14:59,253] Trial 19 pruned.
[I 2023-03-29 13:15:00,415] Trial 20 pruned.
[I 2023-03-29 13:15:03,067] Trial 21 pruned.
[I 2023-03-29 13:15:05,534] Trial 22 pruned.
[I 2023-03-29 13:15:54,543] Trial 23 finished with value: 11.526199340820312 and parameters: {'alpha_dropout': 0.03410663510502585, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008354999801810942, 'wd': 0.005596488034834678, 'n_epochs': 80, 'batch_log2': 3}. Best is trial 23 with value: 11.526199340820312.
[I 2023-03-29 13:16:01,945] Trial 24 pruned.
{'alpha_dropout': 0.03410663510502585, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008354999801810942, 'wd': 0.005596488034834678, 'n_epochs': 80, 'batch_log2': 3}
val mse:  11.526
test mse: 37.016
null mse: 64.747
[I 2023-03-29 13:16:02,225] A new study created in memory with name: gnns-graphs-tuning-i_fold=4-20230329T1216Z
[I 2023-03-29 13:16:58,634] Trial 0 finished with value: 24.32501792907715 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 24.32501792907715.
[I 2023-03-29 13:17:56,083] Trial 1 finished with value: 24.896284103393555 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 24.32501792907715.
[I 2023-03-29 13:18:47,851] Trial 2 finished with value: 24.12174415588379 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 24.12174415588379.
[I 2023-03-29 13:18:48,908] Trial 3 pruned.
[I 2023-03-29 13:18:50,304] Trial 4 pruned.
[I 2023-03-29 13:18:58,117] Trial 5 pruned.
[I 2023-03-29 13:19:27,010] Trial 6 finished with value: 29.731237411499023 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 6}. Best is trial 2 with value: 24.12174415588379.
[I 2023-03-29 13:19:33,314] Trial 7 pruned.
[I 2023-03-29 13:19:34,604] Trial 8 pruned.
[I 2023-03-29 13:20:13,430] Trial 9 finished with value: 19.893178939819336 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 9 with value: 19.893178939819336.
[I 2023-03-29 13:20:15,027] Trial 10 pruned.
[I 2023-03-29 13:20:20,585] Trial 11 pruned.
[I 2023-03-29 13:20:54,406] Trial 12 finished with value: 23.904560089111328 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 9 with value: 19.893178939819336.
[I 2023-03-29 13:20:59,775] Trial 13 pruned.
[I 2023-03-29 13:21:06,439] Trial 14 pruned.
[I 2023-03-29 13:21:38,703] Trial 15 finished with value: 21.086040496826172 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 5}. Best is trial 9 with value: 19.893178939819336.
[I 2023-03-29 13:21:39,989] Trial 16 pruned.
[I 2023-03-29 13:21:46,465] Trial 17 pruned.
[I 2023-03-29 13:21:49,167] Trial 18 pruned.
[I 2023-03-29 13:21:50,763] Trial 19 pruned.
[I 2023-03-29 13:22:22,368] Trial 20 finished with value: 18.949317932128906 and parameters: {'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.006668667514169753, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 20 with value: 18.949317932128906.
[I 2023-03-29 13:22:23,410] Trial 21 pruned.
[I 2023-03-29 13:22:25,844] Trial 22 pruned.
[I 2023-03-29 13:22:31,606] Trial 23 pruned.
[I 2023-03-29 13:22:32,763] Trial 24 pruned.
{'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.006668667514169753, 'n_epochs': 50, 'batch_log2': 2}
val mse:  18.949
test mse: 18.008
null mse: 26.643
[I 2023-03-29 13:22:33,040] A new study created in memory with name: gnns-graphs-tuning-i_fold=5-20230329T1222Z
[I 2023-03-29 13:23:29,653] Trial 0 finished with value: 16.867887496948242 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 16.867887496948242.
[I 2023-03-29 13:24:30,354] Trial 1 finished with value: 17.447322845458984 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 16.867887496948242.
[I 2023-03-29 13:25:24,964] Trial 2 finished with value: 12.788671493530273 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 12.788671493530273.
[I 2023-03-29 13:26:18,617] Trial 3 finished with value: 15.68189811706543 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 2 with value: 12.788671493530273.
[I 2023-03-29 13:26:20,098] Trial 4 pruned.
[I 2023-03-29 13:26:43,766] Trial 5 pruned.
[I 2023-03-29 13:26:44,936] Trial 6 pruned.
[I 2023-03-29 13:27:03,872] Trial 7 pruned.
[I 2023-03-29 13:28:17,242] Trial 8 finished with value: 23.552593231201172 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 2 with value: 12.788671493530273.
[I 2023-03-29 13:28:20,610] Trial 9 pruned.
[I 2023-03-29 13:29:11,574] Trial 10 finished with value: 9.873005867004395 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 9.873005867004395.
[I 2023-03-29 13:29:14,117] Trial 11 pruned.
[I 2023-03-29 13:29:52,710] Trial 12 finished with value: 8.406941413879395 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 12 with value: 8.406941413879395.
[I 2023-03-29 13:29:53,891] Trial 13 pruned.
[I 2023-03-29 13:30:28,116] Trial 14 finished with value: 14.134387969970703 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 12 with value: 8.406941413879395.
[I 2023-03-29 13:30:29,198] Trial 15 pruned.
[I 2023-03-29 13:30:36,208] Trial 16 pruned.
[I 2023-03-29 13:31:46,949] Trial 17 finished with value: 10.651728630065918 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 12 with value: 8.406941413879395.
[I 2023-03-29 13:31:48,294] Trial 18 pruned.
[I 2023-03-29 13:31:51,283] Trial 19 pruned.
[I 2023-03-29 13:31:52,596] Trial 20 pruned.
[I 2023-03-29 13:31:55,593] Trial 21 pruned.
[I 2023-03-29 13:31:57,113] Trial 22 pruned.
[I 2023-03-29 13:31:59,944] Trial 23 pruned.
[I 2023-03-29 13:32:02,489] Trial 24 pruned.
{'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 4}
val mse:  8.407
test mse: 29.014
null mse: 46.614
-------------------------------------------------------------------------------
overall mmse:  21.11
null mmse:  36.35
baseline mmse: 35.08
"""
