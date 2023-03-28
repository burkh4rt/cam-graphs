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

# import optuna.visualization.matplotlib as opt_mpl
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


def model_fold(i_fold: int):
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
    # collect results from predicting on the test set of each fold
    results = list()
    for i in range(data.n_folds):
        results.append(model_fold(i))
    pd.concat(results, axis=0).to_csv(
        os.path.join("data", "age_predictions_av1451_only.csv")
    )

"""
[I 2023-03-28 13:35:22,690] A new study created in memory with name: gnns-graphs-tuning-i_fold=0-20230328T1235Z
[I 2023-03-28 13:36:16,428] Trial 0 finished with value: 15.771749496459961 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 15.771749496459961.
[I 2023-03-28 13:37:13,571] Trial 1 finished with value: 17.192148208618164 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 15.771749496459961.
[I 2023-03-28 13:37:58,360] Trial 2 finished with value: 15.44450569152832 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 2 with value: 15.44450569152832.
[I 2023-03-28 13:37:59,398] Trial 3 pruned.
[I 2023-03-28 13:38:00,587] Trial 4 pruned.
[I 2023-03-28 13:38:03,666] Trial 5 pruned.
[I 2023-03-28 13:38:04,739] Trial 6 pruned.
[I 2023-03-28 13:38:05,842] Trial 7 pruned.
[I 2023-03-28 13:39:10,556] Trial 8 finished with value: 18.677391052246094 and parameters: {'alpha_dropout': 0.08631034258755936, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.007566455605042577, 'wd': 0.0064118189664166105, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 2 with value: 15.44450569152832.
[I 2023-03-28 13:39:11,930] Trial 9 pruned.
[I 2023-03-28 13:39:13,316] Trial 10 pruned.
[I 2023-03-28 13:40:08,264] Trial 11 finished with value: 17.69173812866211 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.001947043582971755, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 2 with value: 15.44450569152832.
[I 2023-03-28 13:40:09,630] Trial 12 pruned.
[I 2023-03-28 13:40:10,760] Trial 13 pruned.
[I 2023-03-28 13:40:15,576] Trial 14 pruned.
[I 2023-03-28 13:40:46,101] Trial 15 finished with value: 13.927699089050293 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 15 with value: 13.927699089050293.
[I 2023-03-28 13:41:15,270] Trial 16 finished with value: 12.673171997070312 and parameters: {'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}. Best is trial 16 with value: 12.673171997070312.
[I 2023-03-28 13:41:17,880] Trial 17 pruned.
[I 2023-03-28 13:41:20,144] Trial 18 pruned.
[I 2023-03-28 13:42:05,327] Trial 19 finished with value: 15.116026878356934 and parameters: {'alpha_dropout': 0.05205269307804442, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.003424997274108889, 'wd': 0.005335876709388094, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 16 with value: 12.673171997070312.
[I 2023-03-28 13:42:07,689] Trial 20 pruned.
[I 2023-03-28 13:42:13,034] Trial 21 pruned.
[I 2023-03-28 13:42:19,743] Trial 22 pruned.
[I 2023-03-28 13:42:21,905] Trial 23 pruned.
[I 2023-03-28 13:42:23,007] Trial 24 pruned.
{'alpha_dropout': 0.09082658859666537, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007049219926652908, 'wd': 0.007640034191754305, 'n_epochs': 60, 'batch_log2': 4}
val mse:  12.673
test mse: 16.365
null mse: 45.881
Permutation explainer: 49it [23:40, 29.59s/it]
[I 2023-03-28 14:06:03,891] A new study created in memory with name: gnns-graphs-tuning-i_fold=1-20230328T1306Z
[I 2023-03-28 14:06:55,428] Trial 0 finished with value: 14.940587997436523 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.940587997436523.
[I 2023-03-28 14:07:47,947] Trial 1 finished with value: 14.165799140930176 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 1 with value: 14.165799140930176.
[I 2023-03-28 14:08:29,650] Trial 2 finished with value: 14.172501564025879 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 1 with value: 14.165799140930176.
[I 2023-03-28 14:08:30,606] Trial 3 pruned.
[I 2023-03-28 14:08:31,736] Trial 4 pruned.
[I 2023-03-28 14:09:07,540] Trial 5 finished with value: 14.689556121826172 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0017964325184672756, 'wd': 0.0020402303379495374, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 1 with value: 14.165799140930176.
[I 2023-03-28 14:09:08,606] Trial 6 pruned.
[I 2023-03-28 14:09:09,641] Trial 7 pruned.
[I 2023-03-28 14:09:12,014] Trial 8 pruned.
[I 2023-03-28 14:09:43,852] Trial 9 finished with value: 12.926647186279297 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.005704595464437947, 'wd': 0.004332656081749642, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 9 with value: 12.926647186279297.
[I 2023-03-28 14:10:22,619] Trial 10 finished with value: 12.445303916931152 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 10 with value: 12.445303916931152.
[I 2023-03-28 14:10:24,676] Trial 11 pruned.
[I 2023-03-28 14:10:25,986] Trial 12 pruned.
[I 2023-03-28 14:10:26,962] Trial 13 pruned.
[I 2023-03-28 14:10:42,988] Trial 14 pruned.
[I 2023-03-28 14:10:56,429] Trial 15 pruned.
[I 2023-03-28 14:10:58,838] Trial 16 pruned.
[I 2023-03-28 14:11:01,783] Trial 17 pruned.
[I 2023-03-28 14:11:16,532] Trial 18 pruned.
[I 2023-03-28 14:12:00,284] Trial 19 finished with value: 17.033388137817383 and parameters: {'alpha_dropout': 0.03410663510502585, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008354999801810942, 'wd': 0.005596488034834678, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 10 with value: 12.445303916931152.
[I 2023-03-28 14:12:01,355] Trial 20 pruned.
[I 2023-03-28 14:12:02,723] Trial 21 pruned.
[I 2023-03-28 14:12:04,040] Trial 22 pruned.
[I 2023-03-28 14:13:12,017] Trial 23 finished with value: 14.335055351257324 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 10 with value: 12.445303916931152.
[I 2023-03-28 14:13:14,064] Trial 24 pruned.
{'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}
val mse:  12.445
test mse: 15.336
null mse: 30.226
[I 2023-03-28 14:13:14,324] A new study created in memory with name: gnns-graphs-tuning-i_fold=2-20230328T1313Z
[I 2023-03-28 14:13:59,956] Trial 0 finished with value: 18.797807693481445 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 18.797807693481445.
[I 2023-03-28 14:14:48,765] Trial 1 finished with value: 20.892770767211914 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 18.797807693481445.
[I 2023-03-28 14:15:26,508] Trial 2 finished with value: 19.387752532958984 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 18.797807693481445.
[I 2023-03-28 14:16:04,198] Trial 3 finished with value: 19.08182144165039 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 0 with value: 18.797807693481445.
[I 2023-03-28 14:16:44,681] Trial 4 finished with value: 17.053224563598633 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 17.053224563598633.
[I 2023-03-28 14:16:45,961] Trial 5 pruned.
[I 2023-03-28 14:16:46,904] Trial 6 pruned.
[I 2023-03-28 14:17:18,992] Trial 7 finished with value: 18.23098373413086 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 4 with value: 17.053224563598633.
[I 2023-03-28 14:17:33,435] Trial 8 pruned.
[I 2023-03-28 14:17:34,568] Trial 9 pruned.
[I 2023-03-28 14:17:35,729] Trial 10 pruned.
[I 2023-03-28 14:17:36,619] Trial 11 pruned.
[I 2023-03-28 14:17:37,719] Trial 12 pruned.
[I 2023-03-28 14:17:38,548] Trial 13 pruned.
[I 2023-03-28 14:17:39,553] Trial 14 pruned.
[I 2023-03-28 14:18:20,722] Trial 15 finished with value: 18.139991760253906 and parameters: {'alpha_dropout': 0.052884574665948016, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.005897137717537036, 'wd': 0.002724747903417664, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 4 with value: 17.053224563598633.
[I 2023-03-28 14:18:21,604] Trial 16 pruned.
[I 2023-03-28 14:18:23,443] Trial 17 pruned.
[I 2023-03-28 14:18:35,479] Trial 18 pruned.
[I 2023-03-28 14:18:49,913] Trial 19 pruned.
[I 2023-03-28 14:18:50,726] Trial 20 pruned.
[I 2023-03-28 14:19:03,971] Trial 21 pruned.
[I 2023-03-28 14:19:05,934] Trial 22 pruned.
[I 2023-03-28 14:19:12,325] Trial 23 pruned.
[I 2023-03-28 14:19:13,183] Trial 24 pruned.
{'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}
val mse:  17.053
test mse: 15.360
null mse: 35.465
[I 2023-03-28 14:19:13,428] A new study created in memory with name: gnns-graphs-tuning-i_fold=3-20230328T1319Z
[I 2023-03-28 14:19:57,177] Trial 0 finished with value: 15.769715309143066 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 15.769715309143066.
[I 2023-03-28 14:20:47,050] Trial 1 finished with value: 19.885969161987305 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 15.769715309143066.
[I 2023-03-28 14:21:27,679] Trial 2 finished with value: 18.000642776489258 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 15.769715309143066.
[I 2023-03-28 14:22:07,532] Trial 3 finished with value: 16.334434509277344 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 0 with value: 15.769715309143066.
[I 2023-03-28 14:22:51,853] Trial 4 finished with value: 15.100428581237793 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 4 with value: 15.100428581237793.
[I 2023-03-28 14:23:11,546] Trial 5 pruned.
[I 2023-03-28 14:23:16,117] Trial 6 pruned.
[I 2023-03-28 14:23:17,108] Trial 7 pruned.
[I 2023-03-28 14:23:18,205] Trial 8 pruned.
[I 2023-03-28 14:23:20,638] Trial 9 pruned.
[I 2023-03-28 14:23:23,142] Trial 10 pruned.
[I 2023-03-28 14:23:25,126] Trial 11 pruned.
[I 2023-03-28 14:23:30,919] Trial 12 pruned.
[I 2023-03-28 14:23:32,566] Trial 13 pruned.
[I 2023-03-28 14:23:38,080] Trial 14 pruned.
[I 2023-03-28 14:23:42,177] Trial 15 pruned.
[I 2023-03-28 14:23:47,609] Trial 16 pruned.
[I 2023-03-28 14:23:48,655] Trial 17 pruned.
[I 2023-03-28 14:23:49,747] Trial 18 pruned.
[I 2023-03-28 14:24:48,271] Trial 19 finished with value: 22.027822494506836 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0075336011098321555, 'wd': 0.008981391573530513, 'n_epochs': 100, 'batch_log2': 4}. Best is trial 4 with value: 15.100428581237793.
[I 2023-03-28 14:24:49,279] Trial 20 pruned.
[I 2023-03-28 14:25:28,417] Trial 21 finished with value: 15.493762016296387 and parameters: {'alpha_dropout': 0.05487337893665861, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.003928597283433409, 'wd': 0.00749026491066844, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 4 with value: 15.100428581237793.
[I 2023-03-28 14:25:30,432] Trial 22 pruned.
[I 2023-03-28 14:25:32,091] Trial 23 pruned.
[I 2023-03-28 14:25:34,666] Trial 24 pruned.
{'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}
val mse:  15.100
test mse: 18.491
null mse: 31.958
"""
