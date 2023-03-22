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


if __name__ == "__main__":
    for i in range(data.n_folds):
        model_fold(i)


"""
IPython could not be loaded!
[I 2023-03-22 16:09:34,432] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230322T1609Z
[I 2023-03-22 16:10:39,358] Trial 0 finished with value: 11.883670806884766 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 11.883670806884766.
[I 2023-03-22 16:11:47,868] Trial 1 finished with value: 12.073939323425293 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 11.883670806884766.
[I 2023-03-22 16:12:43,685] Trial 2 finished with value: 12.329141616821289 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 11.883670806884766.
[I 2023-03-22 16:12:44,903] Trial 3 pruned.
[I 2023-03-22 16:12:46,345] Trial 4 pruned.
{'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}
val mse:  11.884
test mse: 14.330
null mse: 46.149
[I 2023-03-22 16:12:46,636] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230322T1612Z
[I 2023-03-22 16:13:53,800] Trial 0 finished with value: 10.574002265930176 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 10.574002265930176.
[I 2023-03-22 16:15:03,883] Trial 1 finished with value: 10.738912582397461 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 10.574002265930176.
[I 2023-03-22 16:15:59,821] Trial 2 finished with value: 9.057589530944824 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 9.057589530944824.
[I 2023-03-22 16:16:06,284] Trial 3 pruned.
[I 2023-03-22 16:16:07,837] Trial 4 pruned.
{'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}
val mse:  9.058
test mse: 13.116
null mse: 43.882
[I 2023-03-22 16:16:08,162] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230322T1616Z
[I 2023-03-22 16:17:10,248] Trial 0 finished with value: 17.58796501159668 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 17.58796501159668.
[I 2023-03-22 16:18:11,466] Trial 1 finished with value: 18.17790412902832 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 17.58796501159668.
[I 2023-03-22 16:19:00,700] Trial 2 finished with value: 13.553336143493652 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 13.553336143493652.
[I 2023-03-22 16:19:01,822] Trial 3 pruned.
[I 2023-03-22 16:19:03,128] Trial 4 pruned.
{'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}
val mse:  13.553
test mse: 8.014
null mse: 26.468
[I 2023-03-22 16:19:03,390] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230322T1619Z
[I 2023-03-22 16:19:58,915] Trial 0 finished with value: 11.165837287902832 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 11.165837287902832.
[I 2023-03-22 16:20:54,794] Trial 1 finished with value: 10.194421768188477 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 10.194421768188477.
[I 2023-03-22 16:21:10,662] Trial 2 pruned.
[I 2023-03-22 16:21:56,621] Trial 3 finished with value: 7.404685020446777 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 7.404685020446777.
[I 2023-03-22 16:22:44,445] Trial 4 finished with value: 9.252674102783203 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 7.404685020446777.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  7.405
test mse: 13.756
null mse: 33.097
[I 2023-03-22 16:22:44,709] A new study created in memory with name: gnns-graphs-tuning-i_fold=4-20230322T1622Z
[I 2023-03-22 16:23:40,722] Trial 0 finished with value: 20.705242156982422 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 20.705242156982422.
[I 2023-03-22 16:24:38,140] Trial 1 finished with value: 21.06892204284668 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 20.705242156982422.
[I 2023-03-22 16:25:26,340] Trial 2 finished with value: 21.489206314086914 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 20.705242156982422.
[I 2023-03-22 16:26:15,025] Trial 3 finished with value: 18.545806884765625 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 18.545806884765625.
[I 2023-03-22 16:26:17,550] Trial 4 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  18.546
test mse: 9.325
null mse: 30.611
"""
