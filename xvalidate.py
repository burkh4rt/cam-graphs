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
import scipy.stats as sp_stats
import sklearn.metrics as skl_mets

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
criterion = t.nn.L1Loss()
skl_loss = skl_mets.mean_absolute_error


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
    print("val loss:  {:.3f}".format(study.best_value))

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
    print("test loss: {:.3f}".format(loss_test(mdl).detach().numpy()))
    print("null loss: {:.3f}".format(loss_null().detach().numpy()))

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
        "overall loss:  {:.2f} (std err: {:.2f})".format(
            skl_loss(
                results.ages.to_numpy().ravel(),
                results.preds.to_numpy().ravel(),
            ),
            sp_stats.sem(
                results.ages.to_numpy().ravel()
                - results.preds.to_numpy().ravel()
            ),
        ),
    )
    print(
        "null loss:  {:.2f}".format(
            skl_loss(
                results.ages.to_numpy().ravel(),
                results.null_preds.to_numpy().ravel(),
            )
        )
    )
    print(
        "baseline loss: {:.2f}".format(
            skl_loss(
                results.ages.to_numpy().ravel(),
                results.ages.to_numpy().mean()
                * np.ones_like(results.ages.to_numpy()),
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
[I 2023-08-22 10:07:28,116] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230822T1407Z
[I 2023-08-22 10:08:23,439] Trial 0 finished with value: 2.663597822189331 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 2.663597822189331.
[I 2023-08-22 10:09:28,480] Trial 1 finished with value: 2.6627097129821777 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 1 with value: 2.6627097129821777.
[I 2023-08-22 10:10:26,196] Trial 2 finished with value: 2.268239736557007 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 2.268239736557007.
[I 2023-08-22 10:10:27,370] Trial 3 pruned.
[I 2023-08-22 10:11:30,353] Trial 4 finished with value: 2.747192859649658 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 2.268239736557007.
[I 2023-08-22 10:12:15,999] Trial 5 finished with value: 2.6884844303131104 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 2 with value: 2.268239736557007.
[I 2023-08-22 10:12:32,276] Trial 6 pruned.
[I 2023-08-22 10:12:35,049] Trial 7 pruned.
[I 2023-08-22 10:12:42,288] Trial 8 pruned.
[I 2023-08-22 10:12:51,029] Trial 9 pruned.
[I 2023-08-22 10:12:52,795] Trial 10 pruned.
[I 2023-08-22 10:12:54,054] Trial 11 pruned.
[I 2023-08-22 10:12:57,103] Trial 12 pruned.
[I 2023-08-22 10:12:58,168] Trial 13 pruned.
[I 2023-08-22 10:13:01,058] Trial 14 pruned.
[I 2023-08-22 10:13:32,889] Trial 15 finished with value: 2.679811954498291 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 5}. Best is trial 2 with value: 2.268239736557007.
[I 2023-08-22 10:13:35,734] Trial 16 pruned.
[I 2023-08-22 10:14:49,448] Trial 17 finished with value: 2.632927656173706 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 2 with value: 2.268239736557007.
[I 2023-08-22 10:14:50,947] Trial 18 pruned.
[I 2023-08-22 10:15:10,337] Trial 19 pruned.
[I 2023-08-22 10:15:17,278] Trial 20 pruned.
[I 2023-08-22 10:15:18,891] Trial 21 pruned.
[I 2023-08-22 10:15:20,332] Trial 22 pruned.
[I 2023-08-22 10:15:22,167] Trial 23 pruned.
[I 2023-08-22 10:15:24,021] Trial 24 pruned.
{'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}
val loss:  2.268
test loss: 2.515
null loss: 3.503
[I 2023-08-22 10:15:24,295] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230822T1415Z
[I 2023-08-22 10:16:34,867] Trial 0 finished with value: 4.031586170196533 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 4.031586170196533.
[I 2023-08-22 10:17:41,670] Trial 1 finished with value: 4.647918701171875 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 4.031586170196533.
[I 2023-08-22 10:18:39,841] Trial 2 finished with value: 4.047183990478516 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 4.031586170196533.
[I 2023-08-22 10:18:42,478] Trial 3 pruned.
[I 2023-08-22 10:19:45,585] Trial 4 finished with value: 3.549359083175659 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:20:31,179] Trial 5 finished with value: 4.270283222198486 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:20:33,495] Trial 6 pruned.
[I 2023-08-22 10:20:35,054] Trial 7 pruned.
[I 2023-08-22 10:20:57,644] Trial 8 pruned.
[I 2023-08-22 10:20:59,463] Trial 9 pruned.
[I 2023-08-22 10:21:01,267] Trial 10 pruned.
[I 2023-08-22 10:21:02,960] Trial 11 pruned.
[I 2023-08-22 10:21:26,950] Trial 12 pruned.
[I 2023-08-22 10:21:28,061] Trial 13 pruned.
[I 2023-08-22 10:22:06,182] Trial 14 finished with value: 3.6676299571990967 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:22:45,944] Trial 15 finished with value: 3.970517158508301 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 5}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:22:53,621] Trial 16 pruned.
[I 2023-08-22 10:22:55,086] Trial 17 pruned.
[I 2023-08-22 10:23:26,730] Trial 18 finished with value: 3.7379329204559326 and parameters: {'alpha_dropout': 0.05136633570622086, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 20, 'optimizer': 'Adagrad', 'lr': 0.005523143180080194, 'wd': 0.008433216287435114, 'n_epochs': 50, 'batch_log2': 5}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:23:29,544] Trial 19 pruned.
[I 2023-08-22 10:23:30,918] Trial 20 pruned.
[I 2023-08-22 10:24:03,569] Trial 21 finished with value: 3.873809576034546 and parameters: {'alpha_dropout': 0.053183523756224586, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 20, 'optimizer': 'Adagrad', 'lr': 0.006397016993995948, 'wd': 0.008061047491531045, 'n_epochs': 50, 'batch_log2': 5}. Best is trial 4 with value: 3.549359083175659.
[I 2023-08-22 10:24:06,440] Trial 22 pruned.
[I 2023-08-22 10:24:07,444] Trial 23 pruned.
[I 2023-08-22 10:24:11,029] Trial 24 pruned.
{'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}
val loss:  3.549
test loss: 2.973
null loss: 2.854
[I 2023-08-22 10:24:11,305] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230822T1424Z
[I 2023-08-22 10:25:07,006] Trial 0 finished with value: 5.05800199508667 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 5.05800199508667.
[I 2023-08-22 10:26:07,664] Trial 1 finished with value: 4.831550121307373 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 1 with value: 4.831550121307373.
[I 2023-08-22 10:27:00,672] Trial 2 finished with value: 4.808835983276367 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 4.808835983276367.
[I 2023-08-22 10:27:06,553] Trial 3 pruned.
[I 2023-08-22 10:27:13,678] Trial 4 pruned.
[I 2023-08-22 10:27:54,925] Trial 5 finished with value: 4.887206554412842 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 2 with value: 4.808835983276367.
[I 2023-08-22 10:28:00,666] Trial 6 pruned.
[I 2023-08-22 10:28:03,281] Trial 7 pruned.
[I 2023-08-22 10:28:23,370] Trial 8 pruned.
[I 2023-08-22 10:28:24,952] Trial 9 pruned.
[I 2023-08-22 10:29:13,646] Trial 10 finished with value: 4.300812244415283 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 4.300812244415283.
[I 2023-08-22 10:29:14,898] Trial 11 pruned.
[I 2023-08-22 10:29:22,424] Trial 12 pruned.
[I 2023-08-22 10:29:59,019] Trial 13 finished with value: 4.564685344696045 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}. Best is trial 10 with value: 4.300812244415283.
[I 2023-08-22 10:30:00,374] Trial 14 pruned.
[I 2023-08-22 10:30:05,648] Trial 15 pruned.
[I 2023-08-22 10:30:07,031] Trial 16 pruned.
[I 2023-08-22 10:30:09,668] Trial 17 pruned.
[I 2023-08-22 10:30:16,113] Trial 18 pruned.
[I 2023-08-22 10:30:23,290] Trial 19 pruned.
[I 2023-08-22 10:30:25,844] Trial 20 pruned.
[I 2023-08-22 10:30:27,044] Trial 21 pruned.
[I 2023-08-22 10:30:42,675] Trial 22 pruned.
[I 2023-08-22 10:30:44,001] Trial 23 pruned.
{'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}
val loss:  4.301
[I 2023-08-22 10:30:46,529] Trial 24 pruned.
test loss: 2.961
null loss: 5.091
[I 2023-08-22 10:30:46,812] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230822T1430Z
[I 2023-08-22 10:31:42,435] Trial 0 finished with value: 3.1052849292755127 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 3.1052849292755127.
[I 2023-08-22 10:32:52,473] Trial 1 finished with value: 3.2948594093322754 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 3.1052849292755127.
[I 2023-08-22 10:32:55,020] Trial 2 pruned.
[I 2023-08-22 10:33:42,046] Trial 3 finished with value: 3.2933645248413086 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 0 with value: 3.1052849292755127.
[I 2023-08-22 10:34:36,177] Trial 4 finished with value: 3.037322998046875 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 3.037322998046875.
[I 2023-08-22 10:35:14,364] Trial 5 finished with value: 3.015766143798828 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 5 with value: 3.015766143798828.
[I 2023-08-22 10:35:42,976] Trial 6 finished with value: 3.027907371520996 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 6}. Best is trial 5 with value: 3.015766143798828.
[I 2023-08-22 10:35:44,196] Trial 7 pruned.
[I 2023-08-22 10:35:45,463] Trial 8 pruned.
[I 2023-08-22 10:35:52,939] Trial 9 pruned.
[I 2023-08-22 10:35:56,010] Trial 10 pruned.
[I 2023-08-22 10:35:58,229] Trial 11 pruned.
[I 2023-08-22 10:36:04,885] Trial 12 pruned.
[I 2023-08-22 10:36:41,382] Trial 13 finished with value: 2.855605125427246 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}. Best is trial 13 with value: 2.855605125427246.
[I 2023-08-22 10:36:59,276] Trial 14 pruned.
[I 2023-08-22 10:37:00,335] Trial 15 pruned.
[I 2023-08-22 10:37:01,574] Trial 16 pruned.
[I 2023-08-22 10:37:04,071] Trial 17 pruned.
[I 2023-08-22 10:37:10,184] Trial 18 pruned.
[I 2023-08-22 10:37:11,628] Trial 19 pruned.
[I 2023-08-22 10:37:14,329] Trial 20 pruned.
[I 2023-08-22 10:37:17,216] Trial 21 pruned.
[I 2023-08-22 10:37:23,243] Trial 22 pruned.
[I 2023-08-22 10:37:25,625] Trial 23 pruned.
[I 2023-08-22 10:37:26,881] Trial 24 pruned.
{'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}
val loss:  2.856
test loss: 4.734
null loss: 6.160
[I 2023-08-22 10:37:27,252] A new study created in memory with name: gnns-graphs-tuning-i_fold=4-20230822T1437Z
[I 2023-08-22 10:38:23,168] Trial 0 finished with value: 3.7393901348114014 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 3.7393901348114014.
[I 2023-08-22 10:39:21,850] Trial 1 finished with value: 3.812757730484009 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 3.7393901348114014.
[I 2023-08-22 10:40:13,419] Trial 2 finished with value: 3.5244359970092773 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 3.5244359970092773.
[I 2023-08-22 10:40:15,521] Trial 3 pruned.
[I 2023-08-22 10:40:22,476] Trial 4 pruned.
[I 2023-08-22 10:40:25,645] Trial 5 pruned.
[I 2023-08-22 10:40:26,801] Trial 6 pruned.
[I 2023-08-22 10:40:28,059] Trial 7 pruned.
[I 2023-08-22 10:40:46,316] Trial 8 pruned.
[I 2023-08-22 10:40:49,506] Trial 9 pruned.
[I 2023-08-22 10:40:57,674] Trial 10 pruned.
[I 2023-08-22 10:40:58,808] Trial 11 pruned.
[I 2023-08-22 10:41:00,186] Trial 12 pruned.
[I 2023-08-22 10:41:02,295] Trial 13 pruned.
[I 2023-08-22 10:41:03,615] Trial 14 pruned.
[I 2023-08-22 10:41:05,768] Trial 15 pruned.
[I 2023-08-22 10:41:08,362] Trial 16 pruned.
[I 2023-08-22 10:41:09,675] Trial 17 pruned.
[I 2023-08-22 10:41:12,192] Trial 18 pruned.
[I 2023-08-22 10:41:13,519] Trial 19 pruned.
[I 2023-08-22 10:41:16,450] Trial 20 pruned.
[I 2023-08-22 10:41:17,649] Trial 21 pruned.
[I 2023-08-22 10:41:18,899] Trial 22 pruned.
[I 2023-08-22 10:41:21,371] Trial 23 pruned.
[I 2023-08-22 10:41:22,746] Trial 24 pruned.
{'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}
val loss:  3.524
test loss: 2.970
null loss: 4.089
[I 2023-08-22 10:41:23,016] A new study created in memory with name: gnns-graphs-tuning-i_fold=5-20230822T1441Z
[I 2023-08-22 10:42:19,440] Trial 0 finished with value: 2.7693898677825928 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 5}. Best is trial 0 with value: 2.7693898677825928.
[I 2023-08-22 10:43:22,898] Trial 1 finished with value: 3.2686808109283447 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 0 with value: 2.7693898677825928.
[I 2023-08-22 10:43:24,318] Trial 2 pruned.
[I 2023-08-22 10:44:18,102] Trial 3 finished with value: 2.741239309310913 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 3 with value: 2.741239309310913.
[I 2023-08-22 10:44:21,053] Trial 4 pruned.
[I 2023-08-22 10:44:22,715] Trial 5 pruned.
[I 2023-08-22 10:44:23,851] Trial 6 pruned.
[I 2023-08-22 10:44:25,200] Trial 7 pruned.
[I 2023-08-22 10:44:26,683] Trial 8 pruned.
[I 2023-08-22 10:44:35,136] Trial 9 pruned.
[I 2023-08-22 10:44:36,906] Trial 10 pruned.
[I 2023-08-22 10:44:38,191] Trial 11 pruned.
[I 2023-08-22 10:44:41,007] Trial 12 pruned.
[I 2023-08-22 10:44:42,596] Trial 13 pruned.
[I 2023-08-22 10:45:21,205] Trial 14 finished with value: 3.4525907039642334 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 3 with value: 2.741239309310913.
[I 2023-08-22 10:45:22,957] Trial 15 pruned.
[I 2023-08-22 10:45:24,240] Trial 16 pruned.
[I 2023-08-22 10:46:01,317] Trial 17 finished with value: 2.9646050930023193 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 6}. Best is trial 3 with value: 2.741239309310913.
[I 2023-08-22 10:46:02,789] Trial 18 pruned.
[I 2023-08-22 10:46:10,869] Trial 19 pruned.
[I 2023-08-22 10:46:12,654] Trial 20 pruned.
[I 2023-08-22 10:46:15,455] Trial 21 pruned.
[I 2023-08-22 10:46:17,157] Trial 22 pruned.
[I 2023-08-22 10:46:49,510] Trial 23 finished with value: 2.355466842651367 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 5}. Best is trial 23 with value: 2.355466842651367.
[I 2023-08-22 10:46:50,935] Trial 24 pruned.
{'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 5}
val loss:  2.355
test loss: 4.031
null loss: 5.227
-------------------------------------------------------------------------------
overall loss:  3.41 (std err: 0.29)
null loss:  4.57
baseline loss: 4.51
"""
