#!/usr/bin/env python3

"""
Tunes hyperparameters with optuna
"""

import datetime
import os
import warnings

import matplotlib.pyplot as plt
import optuna as opt
import optuna.visualization.matplotlib as opt_mpl
import torch as t
import torch_geometric.loader as t_loader

import dataset
import model

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

    opt_mpl.plot_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    plt.show()

    imps = opt.importance.get_param_importances(
        study, evaluator=opt.importance.FanovaImportanceEvaluator(seed=42)
    )
    feats_by_imps = sorted(imps.keys(), key=lambda k: imps[k], reverse=True)
    opt_mpl.plot_contour(study, params=feats_by_imps[:2])
    plt.show()

"""
[I 2023-03-21 15:10:25,821] A new study created in memory with name: gnns-graphs-tuning-20230321T1510Z
[I 2023-03-21 15:11:13,422] Trial 0 finished with value: 19.496726989746094 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 5, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0015227525095137954, 'wd': 0.008675143843171858, 'n_epochs': 80, 'batch_log2': 4}. Best is trial 0 with value: 19.496726989746094.
[I 2023-03-21 15:12:05,161] Trial 1 finished with value: 22.97626495361328 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 5, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.00373818018663584, 'wd': 0.005295088673159155, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 0 with value: 19.496726989746094.
[I 2023-03-21 15:12:09,894] Trial 2 pruned.
[I 2023-03-21 15:12:45,563] Trial 3 finished with value: 16.23769187927246 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.0037415239225603364, 'wd': 0.0010669539286632004, 'n_epochs': 90, 'batch_log2': 3}. Best is trial 3 with value: 16.23769187927246.
[I 2023-03-21 15:13:24,838] Trial 4 finished with value: 17.739498138427734 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 3, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0038053996848046987, 'wd': 0.005248673409660328, 'n_epochs': 80, 'batch_log2': 2}. Best is trial 3 with value: 16.23769187927246.
[I 2023-03-21 15:13:27,493] Trial 5 pruned.
[I 2023-03-21 15:13:51,945] Trial 6 finished with value: 14.731209754943848 and parameters: {'alpha_dropout': 0.038867728968948204, 'gat_heads': 2, 'gat_out_channels': 5, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.002268318024772864, 'wd': 0.008041750109464993, 'n_epochs': 50, 'batch_log2': 4}. Best is trial 6 with value: 14.731209754943848.
[I 2023-03-21 15:14:23,010] Trial 7 finished with value: 22.800344467163086 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007941433120173511, 'wd': 0.0008330420521674947, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 6 with value: 14.731209754943848.
[I 2023-03-21 15:14:24,117] Trial 8 pruned.
[I 2023-03-21 15:14:25,414] Trial 9 pruned.
[I 2023-03-21 15:15:01,420] Trial 10 finished with value: 14.998353004455566 and parameters: {'alpha_dropout': 0.0031429185686734254, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004693446307320668, 'wd': 0.007579956271576183, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 6 with value: 14.731209754943848.
[I 2023-03-21 15:15:45,886] Trial 11 finished with value: 16.468515396118164 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008233048692092031, 'wd': 0.001947043582971755, 'n_epochs': 100, 'batch_log2': 3}. Best is trial 6 with value: 14.731209754943848.
[I 2023-03-21 15:15:47,074] Trial 12 pruned.
[I 2023-03-21 15:15:47,896] Trial 13 pruned.
[I 2023-03-21 15:15:52,912] Trial 14 pruned.
[I 2023-03-21 15:15:54,553] Trial 15 pruned.
[I 2023-03-21 15:16:24,254] Trial 16 finished with value: 18.58037757873535 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 4, 'gat_out_channels': 4, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.0038870205847456227, 'wd': 0.0019465332529585572, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 6 with value: 14.731209754943848.
[I 2023-03-21 15:16:28,837] Trial 17 pruned.
[I 2023-03-21 15:16:30,776] Trial 18 pruned.
[I 2023-03-21 15:16:37,223] Trial 19 pruned.
[I 2023-03-21 15:16:39,020] Trial 20 pruned.
[I 2023-03-21 15:16:40,123] Trial 21 pruned.
[I 2023-03-21 15:16:41,849] Trial 22 pruned.
[I 2023-03-21 15:16:44,395] Trial 23 pruned.
[I 2023-03-21 15:16:45,369] Trial 24 pruned.
[I 2023-03-21 15:16:47,075] Trial 25 pruned.
[I 2023-03-21 15:17:24,947] Trial 26 finished with value: 13.394796371459961 and parameters: {'alpha_dropout': 0.007372829834525724, 'gat_heads': 4, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004483637328763474, 'wd': 0.009890694086974995, 'n_epochs': 60, 'batch_log2': 2}. Best is trial 26 with value: 13.394796371459961.
[I 2023-03-21 15:17:26,148] Trial 27 pruned.
[I 2023-03-21 15:17:28,298] Trial 28 pruned.
[I 2023-03-21 15:17:29,586] Trial 29 pruned.
[I 2023-03-21 15:17:30,749] Trial 30 pruned.
[I 2023-03-21 15:17:31,990] Trial 31 pruned.
[I 2023-03-21 15:17:33,301] Trial 32 pruned.
[I 2023-03-21 15:17:35,099] Trial 33 pruned.
[I 2023-03-21 15:17:36,059] Trial 34 pruned.
[I 2023-03-21 15:17:37,930] Trial 35 pruned.
[I 2023-03-21 15:17:43,727] Trial 36 pruned.
[I 2023-03-21 15:17:44,756] Trial 37 pruned.
[I 2023-03-21 15:17:47,284] Trial 38 pruned.
[I 2023-03-21 15:17:52,064] Trial 39 pruned.
[I 2023-03-21 15:17:53,095] Trial 40 pruned.
[I 2023-03-21 15:17:55,147] Trial 41 pruned.
[I 2023-03-21 15:17:57,692] Trial 42 pruned.
[I 2023-03-21 15:17:58,784] Trial 43 pruned.
[I 2023-03-21 15:17:59,947] Trial 44 pruned.
[I 2023-03-21 15:18:01,331] Trial 45 pruned.
[I 2023-03-21 15:18:02,560] Trial 46 pruned.
[I 2023-03-21 15:18:03,787] Trial 47 pruned.
[I 2023-03-21 15:18:04,875] Trial 48 pruned.
[I 2023-03-21 15:18:19,953] Trial 49 pruned.
[I 2023-03-21 15:18:23,759] Trial 50 pruned.
[I 2023-03-21 15:18:25,734] Trial 51 pruned.
[I 2023-03-21 15:18:27,010] Trial 52 pruned.
[I 2023-03-21 15:18:32,324] Trial 53 pruned.
[I 2023-03-21 15:18:33,502] Trial 54 pruned.
[I 2023-03-21 15:18:35,239] Trial 55 pruned.
[I 2023-03-21 15:18:37,249] Trial 56 pruned.
[I 2023-03-21 15:18:38,966] Trial 57 pruned.
[I 2023-03-21 15:19:19,123] Trial 58 finished with value: 14.088508605957031 and parameters: {'alpha_dropout': 0.06150072266991698, 'gat_heads': 5, 'gat_out_channels': 1, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.007273141668957412, 'wd': 0.007054592431472382, 'n_epochs': 70, 'batch_log2': 2}. Best is trial 26 with value: 13.394796371459961.
[I 2023-03-21 15:19:20,495] Trial 59 pruned.
[I 2023-03-21 15:20:00,079] Trial 60 finished with value: 15.283683776855469 and parameters: {'alpha_dropout': 0.005956597885222122, 'gat_heads': 4, 'gat_out_channels': 2, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0026039988523586244, 'wd': 0.007135649062885912, 'n_epochs': 70, 'batch_log2': 3}. Best is trial 26 with value: 13.394796371459961.
[I 2023-03-21 15:20:01,275] Trial 61 pruned.
[I 2023-03-21 15:20:02,975] Trial 62 pruned.
[I 2023-03-21 15:20:04,274] Trial 63 pruned.
[I 2023-03-21 15:20:10,703] Trial 64 pruned.
[I 2023-03-21 15:20:12,786] Trial 65 pruned.
[I 2023-03-21 15:20:14,235] Trial 66 pruned.
[I 2023-03-21 15:20:21,941] Trial 67 pruned.
[I 2023-03-21 15:20:27,797] Trial 68 pruned.
[I 2023-03-21 15:20:29,355] Trial 69 pruned.
[I 2023-03-21 15:20:31,045] Trial 70 pruned.
[I 2023-03-21 15:20:32,619] Trial 71 pruned.
[I 2023-03-21 15:20:51,892] Trial 72 pruned.
[I 2023-03-21 15:21:07,728] Trial 73 pruned.
[I 2023-03-21 15:21:10,112] Trial 74 pruned.
[I 2023-03-21 15:21:14,793] Trial 75 pruned.
[I 2023-03-21 15:21:16,410] Trial 76 pruned.
[I 2023-03-21 15:21:18,902] Trial 77 pruned.
[I 2023-03-21 15:21:20,300] Trial 78 pruned.
[I 2023-03-21 15:21:21,603] Trial 79 pruned.
[I 2023-03-21 15:21:24,076] Trial 80 pruned.
[I 2023-03-21 15:21:24,933] Trial 81 pruned.
[I 2023-03-21 15:21:38,377] Trial 82 pruned.
[I 2023-03-21 15:21:42,832] Trial 83 pruned.
[I 2023-03-21 15:21:49,548] Trial 84 pruned.
[I 2023-03-21 15:21:50,865] Trial 85 pruned.
[I 2023-03-21 15:21:52,187] Trial 86 pruned.
[I 2023-03-21 15:21:53,353] Trial 87 pruned.
[I 2023-03-21 15:22:09,667] Trial 88 pruned.
[I 2023-03-21 15:22:13,739] Trial 89 pruned.
[I 2023-03-21 15:22:15,916] Trial 90 pruned.
[I 2023-03-21 15:22:48,325] Trial 91 finished with value: 14.259912490844727 and parameters: {'alpha_dropout': 0.011455049762678329, 'gat_heads': 4, 'gat_out_channels': 5, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.001873789438902175, 'wd': 0.006637197325338482, 'n_epochs': 50, 'batch_log2': 3}. Best is trial 26 with value: 13.394796371459961.
[I 2023-03-21 15:22:50,722] Trial 92 pruned.
[I 2023-03-21 15:22:52,114] Trial 93 pruned.
[I 2023-03-21 15:22:53,847] Trial 94 pruned.
[I 2023-03-21 15:22:56,070] Trial 95 pruned.
[I 2023-03-21 15:23:12,363] Trial 96 pruned.
[I 2023-03-21 15:23:14,521] Trial 97 pruned.
[I 2023-03-21 15:23:18,750] Trial 98 pruned.
[I 2023-03-21 15:23:21,572] Trial 99 pruned.
{'alpha_dropout': 0.007372829834525724, 'gat_heads': 4, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.004483637328763474, 'wd': 0.009890694086974995, 'n_epochs': 60, 'batch_log2': 2}
val mse:  13.395
test mse: 10.221
null mse: 29.627
"""
