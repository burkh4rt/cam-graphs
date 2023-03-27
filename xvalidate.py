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
        os.path.join("data", "age_predictions_no_av1451.csv")
    )

"""
[I 2023-03-27 09:54:49,149] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230327T0854Z
[I 2023-03-27 09:55:39,630] Trial 0 finished with value: 17.953411102294922 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 17.953411102294922.
[I 2023-03-27 09:56:35,439] Trial 1 finished with value: 20.14181900024414 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 17.953411102294922.
[I 2023-03-27 09:57:19,844] Trial 2 finished with value: 17.627750396728516 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 17.627750396728516.
[I 2023-03-27 09:57:20,864] Trial 3 pruned.
[I 2023-03-27 09:58:09,261] Trial 4 finished with value: 13.650272369384766 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 13.650272369384766.
[I 2023-03-27 09:58:10,814] Trial 5 pruned.
[I 2023-03-27 09:58:12,969] Trial 6 pruned.
[I 2023-03-27 09:58:51,311] Trial 7 finished with value: 13.190817832946777 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 7 with value: 13.190817832946777.
[I 2023-03-27 09:59:55,505] Trial 8 finished with value: 12.681585311889648 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 8 with value: 12.681585311889648.
[I 2023-03-27 09:59:58,262] Trial 9 pruned.
[I 2023-03-27 09:59:59,664] Trial 10 pruned.
[I 2023-03-27 10:00:00,768] Trial 11 pruned.
[I 2023-03-27 10:00:02,140] Trial 12 pruned.
[I 2023-03-27 10:00:03,104] Trial 13 pruned.
[I 2023-03-27 10:00:04,337] Trial 14 pruned.
[I 2023-03-27 10:00:33,181] Trial 15 finished with value: 14.739439964294434 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}. Best is trial 8 with value: 12.681585311889648.
[I 2023-03-27 10:00:35,777] Trial 16 pruned.
[I 2023-03-27 10:00:36,925] Trial 17 pruned.
[I 2023-03-27 10:00:38,082] Trial 18 pruned.
[I 2023-03-27 10:00:39,242] Trial 19 pruned.
[I 2023-03-27 10:00:40,428] Trial 20 pruned.
[I 2023-03-27 10:00:41,533] Trial 21 pruned.
[I 2023-03-27 10:00:42,914] Trial 22 pruned.
[I 2023-03-27 10:00:44,266] Trial 23 pruned.
[I 2023-03-27 10:00:45,520] Trial 24 pruned.
{'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 3}
val mse:  12.682
test mse: 38.039
null mse: 45.881
Permutation explainer: 49it [58:42, 73.39s/it]
[I 2023-03-27 10:59:29,387] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230327T0959Z
[I 2023-03-27 11:13:10,428] Trial 0 finished with value: 26.39473533630371 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 26.39473533630371.
[I 2023-03-27 11:15:27,498] Trial 1 finished with value: 25.399463653564453 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 25.399463653564453.
[I 2023-03-27 11:16:49,481] Trial 2 finished with value: 23.047746658325195 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 23.047746658325195.
[I 2023-03-27 11:18:13,697] Trial 3 finished with value: 22.618091583251953 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 22.618091583251953.
[I 2023-03-27 11:18:15,874] Trial 4 pruned.
[I 2023-03-27 11:18:18,483] Trial 5 pruned.
[I 2023-03-27 11:18:20,491] Trial 6 pruned.
[I 2023-03-27 11:18:22,471] Trial 7 pruned.
[I 2023-03-27 11:18:52,682] Trial 8 pruned.
[I 2023-03-27 11:18:54,981] Trial 9 pruned.
[I 2023-03-27 11:19:06,769] Trial 10 pruned.
[I 2023-03-27 11:19:08,707] Trial 11 pruned.
[I 2023-03-27 11:19:41,085] Trial 12 pruned.
[I 2023-03-27 11:19:42,921] Trial 13 pruned.
[I 2023-03-27 11:27:20,369] Trial 14 finished with value: 20.71217918395996 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 14 with value: 20.71217918395996.
[I 2023-03-27 11:27:28,665] Trial 15 pruned.
[I 2023-03-27 11:27:38,296] Trial 16 pruned.
[I 2023-03-27 11:29:05,492] Trial 17 finished with value: 23.036617279052734 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 14 with value: 20.71217918395996.
[I 2023-03-27 11:29:08,756] Trial 18 pruned.
[I 2023-03-27 11:29:18,924] Trial 19 pruned.
[I 2023-03-27 11:29:20,682] Trial 20 pruned.
[I 2023-03-27 11:29:22,700] Trial 21 pruned.
[I 2023-03-27 11:29:24,754] Trial 22 pruned.
[I 2023-03-27 11:41:12,193] Trial 23 pruned.
[I 2023-03-27 11:41:15,847] Trial 24 pruned.
{'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}
val mse:  20.712
test mse: 14.232
null mse: 30.226
[I 2023-03-27 11:41:16,086] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230327T1041Z
[I 2023-03-27 11:42:01,390] Trial 0 finished with value: 17.592636108398438 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 17.592636108398438.
[I 2023-03-27 11:42:48,048] Trial 1 finished with value: 18.44434928894043 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 17.592636108398438.
[I 2023-03-27 11:43:24,600] Trial 2 finished with value: 16.958927154541016 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 16.958927154541016.
[I 2023-03-27 11:43:28,662] Trial 3 pruned.
[I 2023-03-27 11:44:08,300] Trial 4 finished with value: 14.889911651611328 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 14.889911651611328.
[I 2023-03-27 11:44:25,573] Trial 5 pruned.
[I 2023-03-27 11:44:26,478] Trial 6 pruned.
[I 2023-03-27 11:44:57,845] Trial 7 finished with value: 14.507129669189453 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 7 with value: 14.507129669189453.
[I 2023-03-27 11:45:48,473] Trial 8 finished with value: 18.08217430114746 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 7 with value: 14.507129669189453.
[I 2023-03-27 11:45:54,058] Trial 9 pruned.
[I 2023-03-27 11:45:59,710] Trial 10 pruned.
[I 2023-03-27 11:46:11,922] Trial 11 pruned.
[I 2023-03-27 11:46:14,072] Trial 12 pruned.
[I 2023-03-27 11:46:14,890] Trial 13 pruned.
[I 2023-03-27 11:46:16,890] Trial 14 pruned.
[I 2023-03-27 11:46:18,533] Trial 15 pruned.
[I 2023-03-27 11:46:20,574] Trial 16 pruned.
[I 2023-03-27 11:46:33,585] Trial 17 pruned.
[I 2023-03-27 11:46:34,561] Trial 18 pruned.
[I 2023-03-27 11:46:40,335] Trial 19 pruned.
[I 2023-03-27 11:46:53,334] Trial 20 pruned.
[I 2023-03-27 11:46:54,277] Trial 21 pruned.
[I 2023-03-27 11:46:55,300] Trial 22 pruned.
[I 2023-03-27 11:47:26,755] Trial 23 finished with value: 15.55915355682373 and parameters: {'alpha_dropout': 0.07436421082261885, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0015793037447983259, 'wd': 0.004469996333750498, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 7 with value: 14.507129669189453.
[I 2023-03-27 11:47:28,549] Trial 24 pruned.
{'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}
val mse:  14.507
test mse: 26.674
null mse: 35.465
[I 2023-03-27 11:47:28,761] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230327T1047Z
[I 2023-03-27 11:48:12,629] Trial 0 finished with value: 32.48313522338867 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 32.48313522338867.
[I 2023-03-27 11:49:02,275] Trial 1 finished with value: 33.119083404541016 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 32.48313522338867.
[I 2023-03-27 11:49:41,913] Trial 2 finished with value: 31.259109497070312 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 31.259109497070312.
[I 2023-03-27 11:49:42,797] Trial 3 pruned.
[I 2023-03-27 11:49:43,871] Trial 4 pruned.
[I 2023-03-27 11:49:45,216] Trial 5 pruned.
[I 2023-03-27 11:49:46,145] Trial 6 pruned.
[I 2023-03-27 11:50:20,759] Trial 7 finished with value: 35.07435989379883 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 2 with value: 31.259109497070312.
[I 2023-03-27 11:50:21,861] Trial 8 pruned.
[I 2023-03-27 11:50:24,281] Trial 9 pruned.
[I 2023-03-27 11:50:30,265] Trial 10 pruned.
[I 2023-03-27 11:51:16,939] Trial 11 finished with value: 33.187503814697266 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.001947043582971755, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 2 with value: 31.259109497070312.
[I 2023-03-27 11:51:18,068] Trial 12 pruned.
[I 2023-03-27 11:51:22,112] Trial 13 pruned.
[I 2023-03-27 11:51:24,255] Trial 14 pruned.
[I 2023-03-27 11:51:25,065] Trial 15 pruned.
[I 2023-03-27 11:51:52,052] Trial 16 finished with value: 30.071182250976562 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 16 with value: 30.071182250976562.
[I 2023-03-27 11:52:41,788] Trial 17 finished with value: 32.48196029663086 and parameters: {'alpha_dropout': 0.06775643618422825, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.007218439642922194, 'wd': 0.003928679928375321, 'n_epochs': 100, 'batch_log2': 2}. Best is trial 16 with value: 30.071182250976562.
[I 2023-03-27 11:52:42,792] Trial 18 pruned.
[I 2023-03-27 11:52:43,801] Trial 19 pruned.
[I 2023-03-27 11:52:46,041] Trial 20 pruned.
[I 2023-03-27 11:52:48,123] Trial 21 pruned.
[I 2023-03-27 11:52:50,421] Trial 22 pruned.
[I 2023-03-27 11:52:52,471] Trial 23 pruned.
[I 2023-03-27 11:52:53,446] Trial 24 pruned.
{'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 3}
val mse:  30.071
test mse: 21.903
null mse: 31.958
"""
