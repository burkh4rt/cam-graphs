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
    # collect results from predicting on the test set of each fold
    results = pd.concat(
        [
            model_fold(i, explain_features=(i == 0), explain_training=(i == 0))
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
[I 2023-03-29 09:22:07,636] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230329T0822Z
[I 2023-03-29 09:23:04,938] Trial 0 finished with value: 19.799001693725586 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 19.799001693725586.
[I 2023-03-29 09:24:08,190] Trial 1 finished with value: 30.341278076171875 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 19.799001693725586.
[I 2023-03-29 09:25:00,525] Trial 2 finished with value: 20.365022659301758 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 19.799001693725586.
[I 2023-03-29 09:25:01,666] Trial 3 pruned.
[I 2023-03-29 09:25:03,038] Trial 4 pruned.
[I 2023-03-29 09:25:45,094] Trial 5 finished with value: 18.95774269104004 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 5 with value: 18.95774269104004.
[I 2023-03-29 09:26:14,934] Trial 6 finished with value: 21.394168853759766 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 5 with value: 18.95774269104004.
[I 2023-03-29 09:26:16,156] Trial 7 pruned.
[I 2023-03-29 09:26:23,289] Trial 8 pruned.
[I 2023-03-29 09:26:26,335] Trial 9 pruned.
[I 2023-03-29 09:27:13,386] Trial 10 finished with value: 14.67839241027832 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 14.67839241027832.
[I 2023-03-29 09:27:15,869] Trial 11 pruned.
[I 2023-03-29 09:27:18,859] Trial 12 pruned.
[I 2023-03-29 09:27:24,262] Trial 13 pruned.
[I 2023-03-29 09:27:31,112] Trial 14 pruned.
[I 2023-03-29 09:27:33,271] Trial 15 pruned.
[I 2023-03-29 09:27:36,109] Trial 16 pruned.
[I 2023-03-29 09:27:38,641] Trial 17 pruned.
[I 2023-03-29 09:27:39,975] Trial 18 pruned.
[I 2023-03-29 09:27:41,609] Trial 19 pruned.
[I 2023-03-29 09:27:48,286] Trial 20 pruned.
[I 2023-03-29 09:27:55,685] Trial 21 pruned.
[I 2023-03-29 09:28:25,880] Trial 22 finished with value: 15.4127836227417 and parameters: {'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.006668667514169753, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 10 with value: 14.67839241027832.
[I 2023-03-29 09:28:27,218] Trial 23 pruned.
[I 2023-03-29 09:28:29,707] Trial 24 pruned.
{'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}
val mse:  14.678
test mse: 11.233
null mse: 22.039
[W 2023-03-29 09:28:30,188] Output figures of this Matplotlib-based `plot_contour` function would be different from those of the Plotly-based `plot_contour`.
Permutation explainer: 43it [22:32, 32.20s/it]
[I 2023-03-29 09:51:03,640] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230329T0851Z
[I 2023-03-29 09:52:01,181] Trial 0 finished with value: 34.484745025634766 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 34.484745025634766.
[I 2023-03-29 09:53:03,892] Trial 1 finished with value: 35.83476638793945 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 34.484745025634766.
[I 2023-03-29 09:56:48,820] Trial 2 finished with value: 35.816654205322266 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 34.484745025634766.
[I 2023-03-29 09:57:39,527] Trial 3 finished with value: 31.758474349975586 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 31.758474349975586.
[I 2023-03-29 09:58:31,907] Trial 4 finished with value: 32.73822784423828 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 31.758474349975586.
[I 2023-03-29 09:58:33,548] Trial 5 pruned.
[I 2023-03-29 09:59:02,936] Trial 6 finished with value: 26.937803268432617 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 6 with value: 26.937803268432617.
[I 2023-03-29 09:59:05,335] Trial 7 pruned.
[I 2023-03-29 09:59:06,726] Trial 8 pruned.
[I 2023-03-29 09:59:27,177] Trial 9 pruned.
[I 2023-03-29 09:59:28,674] Trial 10 pruned.
[I 2023-03-29 09:59:31,086] Trial 11 pruned.
[I 2023-03-29 10:00:07,698] Trial 12 finished with value: 26.92799186706543 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 12 with value: 26.92799186706543.
[I 2023-03-29 10:00:09,856] Trial 13 pruned.
[I 2023-03-29 10:00:11,181] Trial 14 pruned.
[I 2023-03-29 10:00:16,538] Trial 15 pruned.
[I 2023-03-29 10:00:19,294] Trial 16 pruned.
[I 2023-03-29 10:00:25,467] Trial 17 pruned.
[I 2023-03-29 10:00:31,905] Trial 18 pruned.
[I 2023-03-29 10:00:34,842] Trial 19 pruned.
[I 2023-03-29 10:00:36,039] Trial 20 pruned.
[I 2023-03-29 10:00:37,264] Trial 21 pruned.
[I 2023-03-29 10:00:38,680] Trial 22 pruned.
[I 2023-03-29 10:00:40,445] Trial 23 pruned.
[I 2023-03-29 10:00:42,962] Trial 24 pruned.
{'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 3}
val mse:  26.928
test mse: 15.371
null mse: 23.104
[I 2023-03-29 10:00:43,271] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230329T0900Z
[I 2023-03-29 10:01:33,466] Trial 0 finished with value: 23.43537712097168 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 23.43537712097168.
[I 2023-03-29 10:02:31,537] Trial 1 finished with value: 22.903413772583008 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 22.903413772583008.
[I 2023-03-29 10:03:18,589] Trial 2 finished with value: 25.652515411376953 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 22.903413772583008.
[I 2023-03-29 10:03:19,644] Trial 3 pruned.
[I 2023-03-29 10:04:10,134] Trial 4 finished with value: 22.76946258544922 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 22.76946258544922.
[I 2023-03-29 10:04:49,249] Trial 5 finished with value: 26.629955291748047 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 4 with value: 22.76946258544922.
[I 2023-03-29 10:04:50,327] Trial 6 pruned.
[I 2023-03-29 10:05:06,404] Trial 7 pruned.
[I 2023-03-29 10:05:07,669] Trial 8 pruned.
[I 2023-03-29 10:05:09,077] Trial 9 pruned.
[I 2023-03-29 10:05:51,984] Trial 10 finished with value: 21.88863754272461 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 21.88863754272461.
[I 2023-03-29 10:06:07,505] Trial 11 pruned.
[I 2023-03-29 10:06:10,201] Trial 12 pruned.
[I 2023-03-29 10:06:11,162] Trial 13 pruned.
[I 2023-03-29 10:06:13,729] Trial 14 pruned.
[I 2023-03-29 10:06:14,689] Trial 15 pruned.
[I 2023-03-29 10:06:17,234] Trial 16 pruned.
[I 2023-03-29 10:07:43,464] Trial 17 finished with value: 16.812528610229492 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 17 with value: 16.812528610229492.
[I 2023-03-29 10:07:48,198] Trial 18 pruned.
[I 2023-03-29 10:08:15,343] Trial 19 pruned.
[I 2023-03-29 10:08:17,620] Trial 20 pruned.
[I 2023-03-29 10:08:45,552] Trial 21 pruned.
[I 2023-03-29 10:08:49,946] Trial 22 pruned.
[I 2023-03-29 10:08:55,179] Trial 23 pruned.
[I 2023-03-29 10:08:59,807] Trial 24 pruned.
{'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}
val mse:  16.813
test mse: 33.212
null mse: 61.027
[I 2023-03-29 10:09:00,490] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230329T0909Z
[I 2023-03-29 10:20:15,762] Trial 0 finished with value: 21.325254440307617 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 21.325254440307617.
[I 2023-03-29 10:21:56,210] Trial 1 finished with value: 22.16417121887207 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 21.325254440307617.
[I 2023-03-29 10:22:41,332] Trial 2 finished with value: 25.235986709594727 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 21.325254440307617.
[I 2023-03-29 10:23:26,492] Trial 3 finished with value: 19.241823196411133 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:24:15,241] Trial 4 finished with value: 21.2772159576416 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:24:53,321] Trial 5 finished with value: 22.292016983032227 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:24:54,364] Trial 6 pruned.
[I 2023-03-29 10:25:33,484] Trial 7 finished with value: 26.913841247558594 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:25:34,731] Trial 8 pruned.
[I 2023-03-29 10:25:41,798] Trial 9 pruned.
[I 2023-03-29 10:26:23,814] Trial 10 finished with value: 21.73478126525879 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:26:25,997] Trial 11 pruned.
[I 2023-03-29 10:26:27,316] Trial 12 pruned.
[I 2023-03-29 10:26:29,198] Trial 13 pruned.
[I 2023-03-29 10:27:00,308] Trial 14 finished with value: 22.290239334106445 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 19.241823196411133.
[I 2023-03-29 10:27:13,598] Trial 15 pruned.
[I 2023-03-29 10:27:16,142] Trial 16 pruned.
[I 2023-03-29 10:27:17,314] Trial 17 pruned.
[I 2023-03-29 10:27:18,560] Trial 18 pruned.
[I 2023-03-29 10:27:19,735] Trial 19 pruned.
[I 2023-03-29 10:27:26,254] Trial 20 pruned.
[I 2023-03-29 10:27:27,564] Trial 21 pruned.
[I 2023-03-29 10:27:28,805] Trial 22 pruned.
[I 2023-03-29 10:27:44,220] Trial 23 pruned.
[I 2023-03-29 10:27:46,975] Trial 24 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  19.242
test mse: 28.065
null mse: 29.128
[I 2023-03-29 10:27:47,254] A new study created in memory with name: gnns-graphs-tuning-i_fold=4-20230329T0927Z
[I 2023-03-29 10:28:45,051] Trial 0 finished with value: 14.327068328857422 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.327068328857422.
[I 2023-03-29 10:29:47,294] Trial 1 finished with value: 13.686159133911133 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 13.686159133911133.
[I 2023-03-29 10:29:48,561] Trial 2 pruned.
[I 2023-03-29 10:30:40,884] Trial 3 finished with value: 10.39941120147705 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 10.39941120147705.
[I 2023-03-29 10:31:34,475] Trial 4 finished with value: 16.384048461914062 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 10.39941120147705.
[I 2023-03-29 10:31:36,141] Trial 5 pruned.
[I 2023-03-29 10:31:37,325] Trial 6 pruned.
[I 2023-03-29 10:31:38,572] Trial 7 pruned.
[I 2023-03-29 10:31:45,567] Trial 8 pruned.
[I 2023-03-29 10:31:53,180] Trial 9 pruned.
[I 2023-03-29 10:32:01,127] Trial 10 pruned.
[I 2023-03-29 10:32:02,351] Trial 11 pruned.
[I 2023-03-29 10:32:09,723] Trial 12 pruned.
[I 2023-03-29 10:32:47,579] Trial 13 finished with value: 11.848447799682617 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}. Best is trial 3 with value: 10.39941120147705.
[I 2023-03-29 10:32:50,367] Trial 14 pruned.
[I 2023-03-29 10:32:51,454] Trial 15 pruned.
[I 2023-03-29 10:32:52,843] Trial 16 pruned.
[I 2023-03-29 10:33:57,243] Trial 17 finished with value: 16.116989135742188 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 3 with value: 10.39941120147705.
[I 2023-03-29 10:33:59,887] Trial 18 pruned.
[I 2023-03-29 10:34:02,836] Trial 19 pruned.
[I 2023-03-29 10:34:03,894] Trial 20 pruned.
[I 2023-03-29 10:34:34,086] Trial 21 finished with value: 8.700712203979492 and parameters: {'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.006668667514169753, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 21 with value: 8.700712203979492.
[I 2023-03-29 10:34:40,211] Trial 22 pruned.
[I 2023-03-29 10:34:45,518] Trial 23 pruned.
[I 2023-03-29 10:34:51,824] Trial 24 pruned.
{'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001913243885794289, 'wd': 0.006668667514169753, 'n_epochs': 50, 'batch_log2': 2}
val mse:  8.701
test mse: 21.383
null mse: 43.658
-------------------------------------------------------------------------------
overall mmse:  22.21
baseline mmse: 35.08
"""
