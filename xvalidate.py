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


def model_fold(i_fold: int, *, explain_features: bool = False, explain_training: bool = False):
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
    results = list()
    for i in range(data.n_folds):
        results.append(model_fold(i))
    pd.concat(results, axis=0).to_csv(
        os.path.join("data", "age_predictions_av1451_only.csv")
    )

"""
[I 2023-03-28 15:11:57,493] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230328T1411Z
[I 2023-03-28 15:12:49,550] Trial 0 finished with value: 23.04003143310547 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 23.04003143310547.
[I 2023-03-28 15:13:46,928] Trial 1 finished with value: 16.796855926513672 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:14:32,614] Trial 2 finished with value: 20.94245147705078 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:15:19,525] Trial 3 finished with value: 17.679311752319336 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:15:22,350] Trial 4 pruned.
[I 2023-03-28 15:15:59,978] Trial 5 finished with value: 22.99236297607422 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:16:02,231] Trial 6 pruned.
[I 2023-03-28 15:16:07,824] Trial 7 pruned.
[I 2023-03-28 15:16:09,207] Trial 8 pruned.
[I 2023-03-28 15:16:12,011] Trial 9 pruned.
[I 2023-03-28 15:16:54,816] Trial 10 finished with value: 22.18105697631836 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:16:55,925] Trial 11 pruned.
[I 2023-03-28 15:16:58,604] Trial 12 pruned.
[I 2023-03-28 15:17:00,548] Trial 13 pruned.
[I 2023-03-28 15:17:03,026] Trial 14 pruned.
[I 2023-03-28 15:17:17,036] Trial 15 pruned.
[I 2023-03-28 15:17:35,279] Trial 16 pruned.
[I 2023-03-28 15:18:32,025] Trial 17 finished with value: 17.78015899658203 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 1 with value: 16.796855926513672.
[I 2023-03-28 15:18:33,218] Trial 18 pruned.
[I 2023-03-28 15:18:35,892] Trial 19 pruned.
[I 2023-03-28 15:18:36,985] Trial 20 pruned.
[I 2023-03-28 15:19:21,381] Trial 21 finished with value: 15.103049278259277 and parameters: {'alpha_dropout': 0.05487337893665861, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.003928597283433409, 'wd': 0.00749026491066844, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 21 with value: 15.103049278259277.
[I 2023-03-28 15:19:22,873] Trial 22 pruned.
[I 2023-03-28 15:19:24,175] Trial 23 pruned.
[I 2023-03-28 15:19:25,457] Trial 24 pruned.
{'alpha_dropout': 0.05487337893665861, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.003928597283433409, 'wd': 0.00749026491066844, 'n_epochs': 80, 'batch_log2': 4}
val mse:  15.103
test mse: 12.213
null mse: 24.532
[I 2023-03-28 15:19:25,776] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230328T1419Z
[I 2023-03-28 15:20:10,538] Trial 0 finished with value: 38.34558868408203 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 38.34558868408203.
[I 2023-03-28 15:21:00,484] Trial 1 finished with value: 30.411657333374023 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:21:39,701] Trial 2 finished with value: 34.531700134277344 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:21:44,188] Trial 3 pruned.
[I 2023-03-28 15:22:27,190] Trial 4 finished with value: 34.207496643066406 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:22:33,823] Trial 5 pruned.
[I 2023-03-28 15:22:35,670] Trial 6 pruned.
[I 2023-03-28 15:22:37,572] Trial 7 pruned.
[I 2023-03-28 15:22:38,686] Trial 8 pruned.
[I 2023-03-28 15:22:39,861] Trial 9 pruned.
[I 2023-03-28 15:22:42,242] Trial 10 pruned.
[I 2023-03-28 15:22:44,154] Trial 11 pruned.
[I 2023-03-28 15:22:45,323] Trial 12 pruned.
[I 2023-03-28 15:23:14,425] Trial 13 finished with value: 33.90543746948242 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:23:41,161] Trial 14 finished with value: 33.613319396972656 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:24:06,215] Trial 15 finished with value: 31.09841537475586 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:24:11,834] Trial 16 pruned.
[I 2023-03-28 15:24:13,827] Trial 17 pruned.
[I 2023-03-28 15:24:15,907] Trial 18 pruned.
[I 2023-03-28 15:24:18,256] Trial 19 pruned.
[I 2023-03-28 15:24:19,210] Trial 20 pruned.
[I 2023-03-28 15:24:58,104] Trial 21 finished with value: 33.380184173583984 and parameters: {'alpha_dropout': 0.05487337893665861, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.003928597283433409, 'wd': 0.00749026491066844, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 1 with value: 30.411657333374023.
[I 2023-03-28 15:25:00,657] Trial 22 pruned.
[I 2023-03-28 15:25:03,562] Trial 23 pruned.
[I 2023-03-28 15:25:05,570] Trial 24 pruned.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}
val mse:  30.412
test mse: 15.867
null mse: 23.947
[I 2023-03-28 15:25:05,910] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230328T1425Z
[I 2023-03-28 15:25:49,835] Trial 0 finished with value: 14.355062484741211 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.355062484741211.
[I 2023-03-28 15:26:35,213] Trial 1 finished with value: 19.079376220703125 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 14.355062484741211.
[I 2023-03-28 15:26:37,019] Trial 2 pruned.
[I 2023-03-28 15:27:14,158] Trial 3 finished with value: 14.93805980682373 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 0 with value: 14.355062484741211.
[I 2023-03-28 15:27:15,193] Trial 4 pruned.
[I 2023-03-28 15:27:16,426] Trial 5 pruned.
[I 2023-03-28 15:27:39,387] Trial 6 finished with value: 15.652204513549805 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 0 with value: 14.355062484741211.
[I 2023-03-28 15:27:40,273] Trial 7 pruned.
[I 2023-03-28 15:27:41,318] Trial 8 pruned.
[I 2023-03-28 15:27:43,514] Trial 9 pruned.
[I 2023-03-28 15:27:45,771] Trial 10 pruned.
[I 2023-03-28 15:27:50,249] Trial 11 pruned.
[I 2023-03-28 15:27:51,328] Trial 12 pruned.
[I 2023-03-28 15:27:55,495] Trial 13 pruned.
[I 2023-03-28 15:27:57,486] Trial 14 pruned.
[I 2023-03-28 15:27:58,315] Trial 15 pruned.
[I 2023-03-28 15:28:03,471] Trial 16 pruned.
[I 2023-03-28 15:28:05,305] Trial 17 pruned.
[I 2023-03-28 15:28:07,228] Trial 18 pruned.
[I 2023-03-28 15:28:23,483] Trial 19 pruned.
[I 2023-03-28 15:28:35,784] Trial 20 pruned.
[I 2023-03-28 15:28:36,626] Trial 21 pruned.
[I 2023-03-28 15:28:41,076] Trial 22 pruned.
[I 2023-03-28 15:28:43,011] Trial 23 pruned.
[I 2023-03-28 15:28:44,862] Trial 24 pruned.
{'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}
val mse:  14.355
test mse: 34.137
null mse: 53.293
[I 2023-03-28 15:28:45,213] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230328T1428Z
[I 2023-03-28 15:29:35,877] Trial 0 finished with value: 15.10457706451416 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 15.10457706451416.
[I 2023-03-28 15:30:29,183] Trial 1 finished with value: 18.243864059448242 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 15.10457706451416.
[I 2023-03-28 15:31:12,625] Trial 2 finished with value: 11.155305862426758 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 11.155305862426758.
[I 2023-03-28 15:31:14,550] Trial 3 pruned.
[I 2023-03-28 15:31:16,901] Trial 4 pruned.
[I 2023-03-28 15:31:18,386] Trial 5 pruned.
[I 2023-03-28 15:31:19,455] Trial 6 pruned.
[I 2023-03-28 15:31:20,535] Trial 7 pruned.
[I 2023-03-28 15:31:37,487] Trial 8 pruned.
[I 2023-03-28 15:31:38,817] Trial 9 pruned.
[I 2023-03-28 15:31:40,162] Trial 10 pruned.
[I 2023-03-28 15:31:41,205] Trial 11 pruned.
[I 2023-03-28 15:31:42,501] Trial 12 pruned.
[I 2023-03-28 15:31:44,074] Trial 13 pruned.
[I 2023-03-28 15:31:45,492] Trial 14 pruned.
[I 2023-03-28 15:32:17,430] Trial 15 finished with value: 18.96165657043457 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 5, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008362132893302439, 'wd': 0.0086212327742378, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 2 with value: 11.155305862426758.
[I 2023-03-28 15:32:19,389] Trial 16 pruned.
[I 2023-03-28 15:32:20,964] Trial 17 pruned.
[I 2023-03-28 15:32:22,282] Trial 18 pruned.
[I 2023-03-28 15:44:29,733] Trial 19 finished with value: 10.87243938446045 and parameters: {'alpha_dropout': 0.03339601705554111, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.002167986365893898, 'wd': 0.0075064539581311245, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 19 with value: 10.87243938446045.
[I 2023-03-28 15:44:32,043] Trial 20 pruned.
[I 2023-03-28 15:51:46,363] Trial 21 finished with value: 7.1192731857299805 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 21 with value: 7.1192731857299805.
[I 2023-03-28 15:51:47,797] Trial 22 pruned.
[I 2023-03-28 15:51:50,651] Trial 23 pruned.
[I 2023-03-28 15:51:51,942] Trial 24 pruned.
{'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}
val mse:  7.119
test mse: 27.279
null mse: 37.008
"""
