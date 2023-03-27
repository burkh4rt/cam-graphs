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

    if i_fold == 0:
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
    results = list()
    for i in range(data.n_folds):
        results.append(model_fold(i))
    pd.concat(results, axis=0).to_csv(
        os.path.join("data", "age_predictions_av1451_only.csv")
    )

"""
[I 2023-03-27 12:44:27,618] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230327T1144Z
[I 2023-03-27 12:45:18,551] Trial 0 finished with value: 15.771749496459961 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 15.771749496459961.
[I 2023-03-27 12:46:14,830] Trial 1 finished with value: 17.192148208618164 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 15.771749496459961.
[I 2023-03-27 12:46:58,451] Trial 2 finished with value: 15.44450569152832 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 15.44450569152832.
[I 2023-03-27 12:47:44,456] Trial 3 finished with value: 13.191466331481934 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 13.191466331481934.
[I 2023-03-27 12:47:50,406] Trial 4 pruned.
[I 2023-03-27 12:47:51,928] Trial 5 pruned.
[I 2023-03-27 12:47:52,987] Trial 6 pruned.
[I 2023-03-27 12:47:54,074] Trial 7 pruned.
[I 2023-03-27 12:48:00,329] Trial 8 pruned.
[I 2023-03-27 12:48:01,677] Trial 9 pruned.
[I 2023-03-27 12:48:08,499] Trial 10 pruned.
[I 2023-03-27 12:48:13,949] Trial 11 pruned.
[I 2023-03-27 12:48:20,716] Trial 12 pruned.
[I 2023-03-27 12:48:22,641] Trial 13 pruned.
[I 2023-03-27 12:48:23,881] Trial 14 pruned.
[I 2023-03-27 12:48:24,840] Trial 15 pruned.
[I 2023-03-27 12:48:43,352] Trial 16 pruned.
[I 2023-03-27 12:48:44,689] Trial 17 pruned.
[I 2023-03-27 12:48:50,332] Trial 18 pruned.
[I 2023-03-27 12:48:51,993] Trial 19 pruned.
[I 2023-03-27 12:48:54,445] Trial 20 pruned.
[I 2023-03-27 12:48:57,295] Trial 21 pruned.
[I 2023-03-27 12:49:00,280] Trial 22 pruned.
[I 2023-03-27 12:49:02,481] Trial 23 pruned.
[I 2023-03-27 12:49:04,338] Trial 24 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  13.191
test mse: 17.092
null mse: 45.881
Permutation explainer: 49it [16:52, 21.10s/it]
[I 2023-03-27 13:05:57,830] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230327T1205Z
[I 2023-03-27 13:06:49,719] Trial 0 finished with value: 14.411497116088867 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.411497116088867.
[I 2023-03-27 13:07:43,068] Trial 1 finished with value: 12.777750968933105 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 12.777750968933105.
[I 2023-03-27 13:08:25,601] Trial 2 finished with value: 14.824051856994629 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 12.777750968933105.
[I 2023-03-27 13:09:09,012] Trial 3 finished with value: 11.75313663482666 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:09:54,360] Trial 4 finished with value: 14.093667984008789 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:10:13,997] Trial 5 pruned.
[I 2023-03-27 13:10:40,757] Trial 6 finished with value: 12.0731201171875 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:10:45,976] Trial 7 pruned.
[I 2023-03-27 13:10:52,209] Trial 8 pruned.
[I 2023-03-27 13:11:23,711] Trial 9 finished with value: 14.044017791748047 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:12:01,656] Trial 10 finished with value: 13.191535949707031 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:12:15,940] Trial 11 pruned.
[I 2023-03-27 13:12:17,236] Trial 12 pruned.
[I 2023-03-27 13:12:19,155] Trial 13 pruned.
[I 2023-03-27 13:12:21,455] Trial 14 pruned.
[I 2023-03-27 13:12:34,966] Trial 15 pruned.
[I 2023-03-27 13:12:40,988] Trial 16 pruned.
[I 2023-03-27 13:12:46,293] Trial 17 pruned.
[I 2023-03-27 13:12:48,522] Trial 18 pruned.
[I 2023-03-27 13:13:55,171] Trial 19 finished with value: 14.083547592163086 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 3 with value: 11.75313663482666.
[I 2023-03-27 13:13:56,320] Trial 20 pruned.
[I 2023-03-27 13:13:58,582] Trial 21 pruned.
[I 2023-03-27 13:13:59,653] Trial 22 pruned.
[I 2023-03-27 13:14:05,968] Trial 23 pruned.
[I 2023-03-27 13:14:08,716] Trial 24 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  11.753
test mse: 14.727
null mse: 30.226
[I 2023-03-27 13:14:08,937] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230327T1214Z
[I 2023-03-27 13:14:53,325] Trial 0 finished with value: 20.223024368286133 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 20.223024368286133.
[I 2023-03-27 13:15:40,365] Trial 1 finished with value: 16.479969024658203 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 16.479969024658203.
[I 2023-03-27 13:15:42,259] Trial 2 pruned.
[I 2023-03-27 13:15:43,084] Trial 3 pruned.
[I 2023-03-27 13:16:23,630] Trial 4 finished with value: 16.88758659362793 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 16.479969024658203.
[I 2023-03-27 13:16:26,176] Trial 5 pruned.
[I 2023-03-27 13:16:27,116] Trial 6 pruned.
[I 2023-03-27 13:16:59,315] Trial 7 finished with value: 20.72702980041504 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 16.479969024658203.
[I 2023-03-27 13:17:01,382] Trial 8 pruned.
[I 2023-03-27 13:17:03,640] Trial 9 pruned.
[I 2023-03-27 13:17:05,936] Trial 10 pruned.
[I 2023-03-27 13:17:10,329] Trial 11 pruned.
[I 2023-03-27 13:17:11,417] Trial 12 pruned.
[I 2023-03-27 13:17:40,317] Trial 13 finished with value: 18.867164611816406 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}. Best is trial 1 with value: 16.479969024658203.
[I 2023-03-27 13:17:41,369] Trial 14 pruned.
[I 2023-03-27 13:17:45,532] Trial 15 pruned.
[I 2023-03-27 13:17:47,618] Trial 16 pruned.
[I 2023-03-27 13:17:52,397] Trial 17 pruned.
[I 2023-03-27 13:17:53,593] Trial 18 pruned.
[I 2023-03-27 13:17:54,798] Trial 19 pruned.
[I 2023-03-27 13:17:56,805] Trial 20 pruned.
[I 2023-03-27 13:18:56,378] Trial 21 finished with value: 24.6059627532959 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 1 with value: 16.479969024658203.
[I 2023-03-27 13:18:57,416] Trial 22 pruned.
[I 2023-03-27 13:19:01,931] Trial 23 pruned.
[I 2023-03-27 13:19:03,305] Trial 24 pruned.
{'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}
val mse:  16.480
test mse: 19.204
null mse: 35.465
[I 2023-03-27 13:19:03,588] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230327T1219Z
[I 2023-03-27 13:19:47,173] Trial 0 finished with value: 17.656150817871094 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 17.656150817871094.
[I 2023-03-27 13:20:37,876] Trial 1 finished with value: 19.886064529418945 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 17.656150817871094.
[I 2023-03-27 13:21:18,678] Trial 2 finished with value: 18.463512420654297 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 17.656150817871094.
[I 2023-03-27 13:21:23,144] Trial 3 pruned.
[I 2023-03-27 13:22:07,106] Trial 4 finished with value: 16.52557373046875 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 16.52557373046875.
[I 2023-03-27 13:22:08,510] Trial 5 pruned.
[I 2023-03-27 13:22:09,450] Trial 6 pruned.
[I 2023-03-27 13:22:11,457] Trial 7 pruned.
[I 2023-03-27 13:22:12,583] Trial 8 pruned.
[I 2023-03-27 13:22:43,753] Trial 9 finished with value: 16.435028076171875 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 9 with value: 16.435028076171875.
[I 2023-03-27 13:22:45,006] Trial 10 pruned.
[I 2023-03-27 13:22:46,930] Trial 11 pruned.
[I 2023-03-27 13:22:49,309] Trial 12 pruned.
[I 2023-03-27 13:23:18,603] Trial 13 finished with value: 17.45512580871582 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}. Best is trial 9 with value: 16.435028076171875.
[I 2023-03-27 13:23:19,722] Trial 14 pruned.
[I 2023-03-27 13:23:20,847] Trial 15 pruned.
[I 2023-03-27 13:23:22,929] Trial 16 pruned.
[I 2023-03-27 13:23:23,926] Trial 17 pruned.
[I 2023-03-27 13:23:24,934] Trial 18 pruned.
[I 2023-03-27 13:23:50,221] Trial 19 finished with value: 15.715558052062988 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}. Best is trial 19 with value: 15.715558052062988.
[I 2023-03-27 13:23:51,253] Trial 20 pruned.
[I 2023-03-27 13:24:19,992] Trial 21 finished with value: 16.92190170288086 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 19 with value: 15.715558052062988.
[I 2023-03-27 13:24:21,186] Trial 22 pruned.
[I 2023-03-27 13:24:22,312] Trial 23 pruned.
[I 2023-03-27 13:24:24,390] Trial 24 pruned.
{'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}
val mse:  15.716
test mse: 18.925
null mse: 31.958
"""
