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

import dataset
import model

plt.rcParams["figure.constrained_layout.use"] = True
warnings.simplefilter("ignore", category=opt.exceptions.ExperimentalWarning)

t.manual_seed(0)
criterion = t.nn.MSELoss()


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
        dim_penultimate=trial.suggest_int("dim_penultimate", 5, 25, step=5),
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


if __name__ == "__main__":
    study = opt.create_study(
        direction="minimize",
        study_name="-".join(
            [
                os.path.basename(os.path.dirname(__file__)),
                "tuning",
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
    study.optimize(objective, n_trials=100)

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

    explainer = shap.Explainer(
        mdl.as_function_of_x_y,
        array_test,
        feature_names=list(dataset.cols_xy1_ravelled),
    )
    shap_values = explainer(array_test)

    plt.subplots()
    shap.plots.bar(shap_values, show=False)
    plt.savefig(os.path.join("figures", f"shap_{study.study_name}.pdf"))

    plt.subplots()
    shap.summary_plot(shap_values, array_test, plot_type="layered_violin", color='coolwarm', show=False)
    plt.savefig(os.path.join("figures", f"shap_violin_{study.study_name}.pdf"))

"""
[I 2023-03-22 14:26:23,537] A new study created in memory with name: gnns-graphs-tuning-20230322T1426Z
[I 2023-03-22 14:27:34,401] Trial 0 finished with value: 14.574060440063477 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 14.574060440063477.
[I 2023-03-22 14:28:43,772] Trial 1 finished with value: 16.939184188842773 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 14.574060440063477.
[I 2023-03-22 14:29:37,074] Trial 2 finished with value: 17.89527130126953 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 2, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002797064039425238, 'wd': 0.005190920940294755, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 0 with value: 14.574060440063477.
[I 2023-03-22 14:30:32,194] Trial 3 finished with value: 11.824369430541992 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:31:29,537] Trial 4 finished with value: 15.386998176574707 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:31:31,345] Trial 5 pruned.
[I 2023-03-22 14:31:32,708] Trial 6 pruned.
[I 2023-03-22 14:32:17,912] Trial 7 finished with value: 18.69548225402832 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:32:39,375] Trial 8 pruned.
[I 2023-03-22 14:32:41,003] Trial 9 pruned.
[I 2023-03-22 14:32:44,283] Trial 10 pruned.
[I 2023-03-22 14:32:45,597] Trial 11 pruned.
[I 2023-03-22 14:32:47,201] Trial 12 pruned.
[I 2023-03-22 14:32:48,435] Trial 13 pruned.
[I 2023-03-22 14:33:24,395] Trial 14 finished with value: 16.128816604614258 and parameters: {'alpha_dropout': 0.09624472949421113, 'gat_heads': 2, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adagrad', 'lr': 0.006486079005819072, 'wd': 0.005076522329965729, 'n_epochs': 50, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:33:25,627] Trial 15 pruned.
[I 2023-03-22 14:33:47,691] Trial 16 pruned.
[I 2023-03-22 14:33:50,441] Trial 17 pruned.
[I 2023-03-22 14:33:51,839] Trial 18 pruned.
[I 2023-03-22 14:33:58,902] Trial 19 pruned.
[I 2023-03-22 14:34:02,366] Trial 20 pruned.
[I 2023-03-22 14:34:20,953] Trial 21 pruned.
[I 2023-03-22 14:34:22,933] Trial 22 pruned.
[I 2023-03-22 14:34:24,303] Trial 23 pruned.
[I 2023-03-22 14:34:27,160] Trial 24 pruned.
[I 2023-03-22 14:34:29,047] Trial 25 pruned.
[I 2023-03-22 14:34:35,845] Trial 26 pruned.
[I 2023-03-22 14:34:37,496] Trial 27 pruned.
[I 2023-03-22 14:34:39,452] Trial 28 pruned.
[I 2023-03-22 14:34:40,906] Trial 29 pruned.
[I 2023-03-22 14:34:49,024] Trial 30 pruned.
[I 2023-03-22 14:34:52,495] Trial 31 pruned.
[I 2023-03-22 14:34:54,477] Trial 32 pruned.
[I 2023-03-22 14:35:55,118] Trial 33 finished with value: 13.224384307861328 and parameters: {'alpha_dropout': 0.018676211454606746, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.003851655938283125, 'wd': 0.007409974419669613, 'n_epochs': 80, 'batch_log2': 3}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:35:58,216] Trial 34 pruned.
[I 2023-03-22 14:36:06,480] Trial 35 pruned.
[I 2023-03-22 14:36:09,536] Trial 36 pruned.
[I 2023-03-22 14:36:12,134] Trial 37 pruned.
[I 2023-03-22 14:36:13,971] Trial 38 pruned.
[I 2023-03-22 14:37:10,768] Trial 39 finished with value: 13.084428787231445 and parameters: {'alpha_dropout': 0.06150072266991698, 'gat_heads': 5, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007273141668957412, 'wd': 0.007054592431472382, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:37:12,281] Trial 40 pruned.
[I 2023-03-22 14:38:29,915] Trial 41 finished with value: 14.22742748260498 and parameters: {'alpha_dropout': 0.08093611554785136, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.008184656610700978, 'wd': 0.006534642914699876, 'n_epochs': 90, 'batch_log2': 4}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:38:32,746] Trial 42 pruned.
[I 2023-03-22 14:38:52,148] Trial 43 pruned.
[I 2023-03-22 14:38:59,353] Trial 44 pruned.
[I 2023-03-22 14:39:06,106] Trial 45 pruned.
[I 2023-03-22 14:39:07,674] Trial 46 pruned.
[I 2023-03-22 14:39:13,883] Trial 47 pruned.
[I 2023-03-22 14:40:11,216] Trial 48 finished with value: 12.74876594543457 and parameters: {'alpha_dropout': 0.003524313873108278, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.0023928483161755687, 'wd': 0.005263844853724547, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:40:33,591] Trial 49 pruned.
[I 2023-03-22 14:40:40,074] Trial 50 pruned.
[I 2023-03-22 14:40:48,229] Trial 51 pruned.
[I 2023-03-22 14:40:51,259] Trial 52 pruned.
[I 2023-03-22 14:40:54,263] Trial 53 pruned.
[I 2023-03-22 14:40:56,984] Trial 54 pruned.
[I 2023-03-22 14:41:49,721] Trial 55 finished with value: 15.762273788452148 and parameters: {'alpha_dropout': 0.03559726786512616, 'gat_heads': 4, 'gat_out_channels': 1, 'dim_penultimate': 5, 'optimizer': 'Adagrad', 'lr': 0.008699145256099066, 'wd': 0.007066212807862235, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:41:56,562] Trial 56 pruned.
[I 2023-03-22 14:41:58,001] Trial 57 pruned.
[I 2023-03-22 14:41:59,485] Trial 58 pruned.
[I 2023-03-22 14:42:02,081] Trial 59 pruned.
[I 2023-03-22 14:42:04,007] Trial 60 pruned.
[I 2023-03-22 14:42:05,450] Trial 61 pruned.
[I 2023-03-22 14:42:08,513] Trial 62 pruned.
[I 2023-03-22 14:42:10,074] Trial 63 pruned.
[I 2023-03-22 14:42:13,326] Trial 64 pruned.
[I 2023-03-22 14:42:14,622] Trial 65 pruned.
[I 2023-03-22 14:42:16,111] Trial 66 pruned.
[I 2023-03-22 14:42:17,887] Trial 67 pruned.
[I 2023-03-22 14:42:21,104] Trial 68 pruned.
[I 2023-03-22 14:43:12,219] Trial 69 pruned.
[I 2023-03-22 14:43:15,737] Trial 70 pruned.
[I 2023-03-22 14:43:17,410] Trial 71 pruned.
[I 2023-03-22 14:43:20,827] Trial 72 pruned.
[I 2023-03-22 14:43:22,351] Trial 73 pruned.
[I 2023-03-22 14:43:25,649] Trial 74 pruned.
[I 2023-03-22 14:43:29,542] Trial 75 pruned.
[I 2023-03-22 14:43:31,156] Trial 76 pruned.
[I 2023-03-22 14:43:32,524] Trial 77 pruned.
[I 2023-03-22 14:43:57,707] Trial 78 pruned.
[I 2023-03-22 14:44:00,978] Trial 79 pruned.
[I 2023-03-22 14:44:10,496] Trial 80 pruned.
[I 2023-03-22 14:44:13,553] Trial 81 pruned.
[I 2023-03-22 14:44:22,456] Trial 82 pruned.
[I 2023-03-22 14:44:30,585] Trial 83 pruned.
[I 2023-03-22 14:44:39,047] Trial 84 pruned.
[I 2023-03-22 14:44:47,116] Trial 85 pruned.
[I 2023-03-22 14:44:48,835] Trial 86 pruned.
[I 2023-03-22 14:45:40,440] Trial 87 finished with value: 15.888378143310547 and parameters: {'alpha_dropout': 0.03881699262065219, 'gat_heads': 4, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.009650715074415228, 'wd': 0.00906297135536503, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 3 with value: 11.824369430541992.
[I 2023-03-22 14:45:42,035] Trial 88 pruned.
[I 2023-03-22 14:45:43,647] Trial 89 pruned.
[I 2023-03-22 14:45:46,664] Trial 90 pruned.
[I 2023-03-22 14:45:55,023] Trial 91 pruned.
[I 2023-03-22 14:45:58,551] Trial 92 pruned.
[I 2023-03-22 14:46:01,071] Trial 93 pruned.
[I 2023-03-22 14:46:02,768] Trial 94 pruned.
[I 2023-03-22 14:46:10,431] Trial 95 pruned.
[I 2023-03-22 14:46:31,365] Trial 96 pruned.
[I 2023-03-22 14:46:32,947] Trial 97 pruned.
[I 2023-03-22 14:46:40,738] Trial 98 pruned.
[I 2023-03-22 14:46:42,527] Trial 99 pruned.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}
val mse:  11.824
test mse: 10.835
null mse: 29.627
"""
