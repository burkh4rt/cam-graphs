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
        os.path.join("data", "age_predictions.csv")
    )

"""
[I 2023-03-24 11:20:04,490] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230324T1120Z
[I 2023-03-24 11:20:55,720] Trial 0 finished with value: 11.035832405090332 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 11.035832405090332.
[I 2023-03-24 11:21:51,695] Trial 1 finished with value: 12.510648727416992 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 11.035832405090332.
[I 2023-03-24 11:22:36,121] Trial 2 finished with value: 9.171838760375977 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 9.171838760375977.
[I 2023-03-24 11:22:41,181] Trial 3 pruned.
[I 2023-03-24 11:22:47,189] Trial 4 pruned.
[I 2023-03-24 11:22:48,707] Trial 5 pruned.
[I 2023-03-24 11:23:15,629] Trial 6 finished with value: 11.308069229125977 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 2 with value: 9.171838760375977.
[I 2023-03-24 11:23:17,814] Trial 7 pruned.
[I 2023-03-24 11:23:19,096] Trial 8 pruned.
[I 2023-03-24 11:23:53,209] Trial 9 finished with value: 9.634523391723633 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 2 with value: 9.171838760375977.
[I 2023-03-24 11:24:00,139] Trial 10 pruned.
[I 2023-03-24 11:24:01,239] Trial 11 pruned.
[I 2023-03-24 11:24:02,586] Trial 12 pruned.
[I 2023-03-24 11:24:03,545] Trial 13 pruned.
[I 2023-03-24 11:24:04,771] Trial 14 pruned.
[I 2023-03-24 11:24:06,702] Trial 15 pruned.
[I 2023-03-24 11:24:39,123] Trial 16 finished with value: 9.473078727722168 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 2 with value: 9.171838760375977.
[I 2023-03-24 11:24:40,259] Trial 17 pruned.
[I 2023-03-24 11:24:41,842] Trial 18 pruned.
[I 2023-03-24 11:24:47,769] Trial 19 pruned.
[I 2023-03-24 11:25:11,464] Trial 20 pruned.
[I 2023-03-24 11:25:12,843] Trial 21 pruned.
[I 2023-03-24 11:25:16,037] Trial 22 pruned.
[I 2023-03-24 11:25:17,420] Trial 23 pruned.
[I 2023-03-24 11:26:25,343] Trial 24 finished with value: 15.13149356842041 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 2 with value: 9.171838760375977.
{'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}
val mse:  9.172
test mse: 18.438
null mse: 45.881
Permutation explainer: 49it [21:25, 26.77s/it]
[I 2023-03-24 11:47:51,141] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230324T1147Z
[I 2023-03-24 11:48:44,325] Trial 0 finished with value: 17.639244079589844 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 17.639244079589844.
[I 2023-03-24 11:49:39,784] Trial 1 finished with value: 12.150362968444824 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 12.150362968444824.
[I 2023-03-24 11:50:23,721] Trial 2 finished with value: 13.994665145874023 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 12.150362968444824.
[I 2023-03-24 11:51:08,138] Trial 3 finished with value: 11.369157791137695 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 11.369157791137695.
[I 2023-03-24 11:51:09,271] Trial 4 pruned.
[I 2023-03-24 11:51:44,296] Trial 5 finished with value: 15.62009048461914 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 11.369157791137695.
[I 2023-03-24 11:51:49,641] Trial 6 pruned.
[I 2023-03-24 11:52:04,011] Trial 7 pruned.
[I 2023-03-24 11:52:05,189] Trial 8 pruned.
[I 2023-03-24 11:52:06,460] Trial 9 pruned.
[I 2023-03-24 11:52:09,029] Trial 10 pruned.
[I 2023-03-24 11:52:23,366] Trial 11 pruned.
[I 2023-03-24 11:52:25,868] Trial 12 pruned.
[I 2023-03-24 11:52:30,647] Trial 13 pruned.
[I 2023-03-24 11:52:36,351] Trial 14 pruned.
[I 2023-03-24 11:52:38,259] Trial 15 pruned.
[I 2023-03-24 11:52:39,453] Trial 16 pruned.
[I 2023-03-24 11:52:40,523] Trial 17 pruned.
[I 2023-03-24 11:52:45,308] Trial 18 pruned.
[I 2023-03-24 11:52:46,376] Trial 19 pruned.
[I 2023-03-24 11:52:51,913] Trial 20 pruned.
[I 2023-03-24 11:53:25,258] Trial 21 finished with value: 12.973318099975586 and parameters: {'alpha_dropout': 0.05410196567651246, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.004531908314837257, 'wd': 0.0006243444496059576, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 3 with value: 11.369157791137695.
[I 2023-03-24 11:53:26,196] Trial 22 pruned.
[I 2023-03-24 11:53:28,124] Trial 23 pruned.
[I 2023-03-24 11:54:35,411] Trial 24 finished with value: 14.994529724121094 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 3 with value: 11.369157791137695.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  11.369
test mse: 9.275
null mse: 30.226
[I 2023-03-24 11:54:35,698] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230324T1154Z
[I 2023-03-24 11:55:20,536] Trial 0 finished with value: 12.40731143951416 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 12.40731143951416.
[I 2023-03-24 11:56:07,835] Trial 1 finished with value: 14.507713317871094 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 12.40731143951416.
[I 2023-03-24 11:56:45,910] Trial 2 finished with value: 12.326193809509277 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 12.326193809509277.
[I 2023-03-24 11:57:23,422] Trial 3 finished with value: 12.06436538696289 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 12.06436538696289.
[I 2023-03-24 11:58:04,703] Trial 4 finished with value: 13.5720853805542 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 12.06436538696289.
[I 2023-03-24 11:58:11,174] Trial 5 pruned.
[I 2023-03-24 11:58:34,416] Trial 6 finished with value: 12.089377403259277 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 3 with value: 12.06436538696289.
[I 2023-03-24 11:58:35,349] Trial 7 pruned.
[I 2023-03-24 11:59:30,034] Trial 8 finished with value: 13.410954475402832 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 3 with value: 12.06436538696289.
[I 2023-03-24 11:59:32,296] Trial 9 pruned.
[I 2023-03-24 11:59:48,095] Trial 10 pruned.
[I 2023-03-24 11:59:48,973] Trial 11 pruned.
[I 2023-03-24 11:59:50,049] Trial 12 pruned.
[I 2023-03-24 12:00:01,565] Trial 13 pruned.
[I 2023-03-24 12:00:02,589] Trial 14 pruned.
[I 2023-03-24 12:00:03,415] Trial 15 pruned.
[I 2023-03-24 12:00:04,440] Trial 16 pruned.
[I 2023-03-24 12:00:05,389] Trial 17 pruned.
[I 2023-03-24 12:00:06,346] Trial 18 pruned.
[I 2023-03-24 12:00:08,330] Trial 19 pruned.
[I 2023-03-24 12:00:10,262] Trial 20 pruned.
[I 2023-03-24 12:00:12,610] Trial 21 pruned.
[I 2023-03-24 12:00:14,441] Trial 22 pruned.
[I 2023-03-24 12:00:16,598] Trial 23 pruned.
[I 2023-03-24 12:00:29,498] Trial 24 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  12.064
test mse: 17.089
null mse: 35.465
[I 2023-03-24 12:00:29,778] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230324T1200Z
[I 2023-03-24 12:01:13,909] Trial 0 finished with value: 18.838003158569336 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:02:04,372] Trial 1 finished with value: 19.82958221435547 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:02:05,397] Trial 2 pruned.
[I 2023-03-24 12:02:45,834] Trial 3 finished with value: 19.114412307739258 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:03:30,576] Trial 4 finished with value: 20.838987350463867 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:03:37,610] Trial 5 pruned.
[I 2023-03-24 12:04:00,612] Trial 6 finished with value: 19.537612915039062 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:04:01,689] Trial 7 pruned.
[I 2023-03-24 12:04:02,850] Trial 8 pruned.
[I 2023-03-24 12:04:34,113] Trial 9 finished with value: 19.216472625732422 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 0 with value: 18.838003158569336.
[I 2023-03-24 12:04:35,369] Trial 10 pruned.
[I 2023-03-24 12:04:36,331] Trial 11 pruned.
[I 2023-03-24 12:04:42,121] Trial 12 pruned.
[I 2023-03-24 12:05:11,152] Trial 13 finished with value: 15.94289493560791 and parameters: {'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}. Best is trial 13 with value: 15.94289493560791.
[I 2023-03-24 12:05:16,710] Trial 14 pruned.
[I 2023-03-24 12:05:18,352] Trial 15 pruned.
[I 2023-03-24 12:05:19,444] Trial 16 pruned.
[I 2023-03-24 12:05:20,568] Trial 17 pruned.
[I 2023-03-24 12:05:21,422] Trial 18 pruned.
[I 2023-03-24 12:05:26,533] Trial 19 pruned.
[I 2023-03-24 12:05:28,679] Trial 20 pruned.
[I 2023-03-24 12:05:29,693] Trial 21 pruned.
[I 2023-03-24 12:05:31,669] Trial 22 pruned.
[I 2023-03-24 12:05:37,544] Trial 23 pruned.
[I 2023-03-24 12:05:42,598] Trial 24 pruned.
{'alpha_dropout': 0.041741100314877905, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.0056691155956902954, 'wd': 0.007059887693062261, 'n_epochs': 70, 'batch_log2': 4}
val mse:  15.943
test mse: 14.418
null mse: 31.958
"""
