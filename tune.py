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
        gat_heads=trial.suggest_int("gat_heads", 1, 3),
        gat_out_channels=trial.suggest_int("gat_out_channels", 1, 3),
        dim_penultimate=trial.suggest_int("dim_penultimate", 5, 25, step=5),
    )
    opt_type = trial.suggest_categorical("optimizer", ["Adagrad", "Adam"])
    optimizer = getattr(t.optim, opt_type)(
        mdl.parameters(),
        lr=trial.suggest_float("lr", 1e-4, 1e-2),
        weight_decay=trial.suggest_float("wd", 1e-5, 1e-4),
    )

    for epoch in range(trial.suggest_int("n_epochs", 5, 50, step=5)):
        mdl.train()
        for data in iter(
            t_loader.DataLoader(
                dataset.data_train,
                batch_size=2 ** trial.suggest_int("batch_log2", 10, 13),
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


"""
[I 2023-03-30 09:50:46,950] A new study created in memory with name: gnns-graphs-tuning-20230330T0850Z
[I 2023-03-30 10:00:47,967] Trial 0 finished with value: 4.409603595733643 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0006750277604651748, 'wd': 8.795585311974417e-05, 'n_epochs': 35, 'batch_log2': 12}. Best is trial 0 with value: 4.409603595733643.
[I 2023-03-30 10:08:50,396] Trial 1 finished with value: 4.016777515411377 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.0031119982052994237, 'wd': 5.722807884690141e-05, 'n_epochs': 25, 'batch_log2': 11}. Best is trial 1 with value: 4.016777515411377.
[I 2023-03-30 10:10:06,961] Trial 2 finished with value: 4.003558158874512 and parameters: {'alpha_dropout': 0.06118528947223795, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.0020767704433677613, 'wd': 5.628109945722505e-05, 'n_epochs': 30, 'batch_log2': 10}. Best is trial 2 with value: 4.003558158874512.
[I 2023-03-30 10:11:30,244] Trial 3 finished with value: 4.002810001373291 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.00311567631481637, 'wd': 1.879049026057455e-05, 'n_epochs': 35, 'batch_log2': 11}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:18:00,561] Trial 4 pruned.
[I 2023-03-30 10:19:34,543] Trial 5 finished with value: 4.110290050506592 and parameters: {'alpha_dropout': 0.09695846277645587, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0009760757703140031, 'wd': 2.763845761772307e-05, 'n_epochs': 5, 'batch_log2': 11}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:19:58,221] Trial 6 finished with value: 4.226093292236328 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.0014951498272501504, 'wd': 8.219772826786358e-05, 'n_epochs': 5, 'batch_log2': 13}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:20:50,370] Trial 7 finished with value: 4.008057594299316 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007735576432190864, 'wd': 1.6664018656068134e-05, 'n_epochs': 20, 'batch_log2': 10}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:21:15,370] Trial 8 pruned.
[I 2023-03-30 10:21:54,769] Trial 9 pruned.
[I 2023-03-30 10:22:21,096] Trial 10 pruned.
[I 2023-03-30 10:22:38,662] Trial 11 pruned.
[I 2023-03-30 10:23:56,720] Trial 12 finished with value: 4.049323081970215 and parameters: {'alpha_dropout': 0.08074401551640625, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 5, 'optimizer': 'Adam', 'lr': 0.008198346182632682, 'wd': 8.746575249307091e-05, 'n_epochs': 5, 'batch_log2': 12}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:24:06,336] Trial 13 pruned.
[I 2023-03-30 10:24:22,918] Trial 14 pruned.
[I 2023-03-30 10:24:32,074] Trial 15 pruned.
[I 2023-03-30 10:24:57,442] Trial 16 pruned.
[I 2023-03-30 10:25:06,874] Trial 17 pruned.
[I 2023-03-30 10:25:54,825] Trial 18 pruned.
[I 2023-03-30 10:27:10,983] Trial 19 pruned.
[I 2023-03-30 10:27:24,228] Trial 20 finished with value: 4.458313465118408 and parameters: {'alpha_dropout': 0.06420316461542878, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.001104568274373718, 'wd': 6.971515921972503e-05, 'n_epochs': 5, 'batch_log2': 10}. Best is trial 3 with value: 4.002810001373291.
[I 2023-03-30 10:27:54,631] Trial 21 pruned.
[I 2023-03-30 10:28:43,243] Trial 22 pruned.
[I 2023-03-30 10:37:02,103] Trial 23 finished with value: 4.014516353607178 and parameters: {'alpha_dropout': 0.07948113035416485, 'gat_heads': 2, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0028796463881644724, 'wd': 1.2188436978830845e-05, 'n_epochs': 35, 'batch_log2': 10}. Best is trial 3 with value: 4.002810001373291.
{'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.00311567631481637, 'wd': 1.879049026057455e-05, 'n_epochs': 35, 'batch_log2': 11}
val mse:  4.003
[I 2023-03-30 10:38:20,184] Trial 24 pruned.
test mse: 3.890
null mse: 4.141
"""
