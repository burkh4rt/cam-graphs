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
[I 2023-03-30 11:35:53,698] A new study created in memory with name: gnns-graphs-tuning-20230330T1035Z
[I 2023-03-30 11:45:49,616] Trial 0 finished with value: 4.555659770965576 and parameters: {'alpha_dropout': 0.03745401188473625, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0006750277604651748, 'wd': 8.795585311974417e-05, 'n_epochs': 35, 'batch_log2': 12}. Best is trial 0 with value: 4.555659770965576.
[I 2023-03-30 11:52:45,657] Trial 1 finished with value: 3.9697775840759277 and parameters: {'alpha_dropout': 0.0020584494295802446, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 10, 'optimizer': 'Adam', 'lr': 0.0031119982052994237, 'wd': 5.722807884690141e-05, 'n_epochs': 25, 'batch_log2': 11}. Best is trial 1 with value: 3.9697775840759277.
[I 2023-03-30 11:52:50,663] Trial 2 pruned.
[I 2023-03-30 11:54:10,804] Trial 3 finished with value: 3.9550933837890625 and parameters: {'alpha_dropout': 0.06075448519014384, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adagrad', 'lr': 0.00311567631481637, 'wd': 1.879049026057455e-05, 'n_epochs': 35, 'batch_log2': 11}. Best is trial 3 with value: 3.9550933837890625.
[I 2023-03-30 12:00:44,416] Trial 4 finished with value: 3.9956910610198975 and parameters: {'alpha_dropout': 0.012203823484477884, 'gat_heads': 2, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.0031859396532851686, 'wd': 5.680612190600298e-05, 'n_epochs': 30, 'batch_log2': 10}. Best is trial 3 with value: 3.9550933837890625.
[I 2023-03-30 12:01:51,375] Trial 5 pruned.
[I 2023-03-30 12:02:00,144] Trial 6 pruned.
[I 2023-03-30 12:02:48,087] Trial 7 finished with value: 3.960188388824463 and parameters: {'alpha_dropout': 0.07722447692966575, 'gat_heads': 1, 'gat_out_channels': 1, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.007735576432190864, 'wd': 1.6664018656068134e-05, 'n_epochs': 20, 'batch_log2': 10}. Best is trial 3 with value: 3.9550933837890625.
[I 2023-03-30 12:03:11,128] Trial 8 pruned.
[I 2023-03-30 12:04:45,462] Trial 9 finished with value: 3.954624652862549 and parameters: {'alpha_dropout': 0.01195942459383017, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 15, 'optimizer': 'Adagrad', 'lr': 0.0052750550108817415, 'wd': 4.847869165226947e-05, 'n_epochs': 5, 'batch_log2': 10}. Best is trial 9 with value: 3.954624652862549.
[I 2023-03-30 12:05:36,828] Trial 10 pruned.
[I 2023-03-30 12:08:51,106] Trial 11 finished with value: 3.9517905712127686 and parameters: {'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008056353561301233, 'wd': 2.6791305299743224e-05, 'n_epochs': 45, 'batch_log2': 12}. Best is trial 11 with value: 3.9517905712127686.
[I 2023-03-30 12:09:48,450] Trial 12 pruned.
[I 2023-03-30 12:10:10,023] Trial 13 pruned.
[I 2023-03-30 12:10:18,603] Trial 14 pruned.
[I 2023-03-30 12:10:23,116] Trial 15 pruned.
[I 2023-03-30 12:10:31,913] Trial 16 pruned.
[I 2023-03-30 12:10:54,218] Trial 17 pruned.
[I 2023-03-30 12:11:59,272] Trial 18 finished with value: 4.029176235198975 and parameters: {'alpha_dropout': 0.03677831327192532, 'gat_heads': 2, 'gat_out_channels': 2, 'dim_penultimate': 15, 'optimizer': 'Adam', 'lr': 0.003275722643220185, 'wd': 2.678666593598688e-05, 'n_epochs': 5, 'batch_log2': 12}. Best is trial 11 with value: 3.9517905712127686.
[I 2023-03-30 12:12:43,453] Trial 19 pruned.
[I 2023-03-30 12:13:29,641] Trial 20 pruned.
[I 2023-03-30 12:27:21,721] Trial 21 finished with value: 3.9629123210906982 and parameters: {'alpha_dropout': 0.009310276780589922, 'gat_heads': 3, 'gat_out_channels': 3, 'dim_penultimate': 20, 'optimizer': 'Adam', 'lr': 0.007286961220815371, 'wd': 9.073992339573195e-05, 'n_epochs': 45, 'batch_log2': 13}. Best is trial 11 with value: 3.9517905712127686.
[I 2023-03-30 12:27:31,685] Trial 22 pruned.
[I 2023-03-30 12:30:00,096] Trial 23 pruned.
[I 2023-03-30 12:30:25,165] Trial 24 pruned.
{'alpha_dropout': 0.028975145291376805, 'gat_heads': 1, 'gat_out_channels': 3, 'dim_penultimate': 25, 'optimizer': 'Adam', 'lr': 0.008056353561301233, 'wd': 2.6791305299743224e-05, 'n_epochs': 45, 'batch_log2': 12}
val mse:  3.952
test mse: 3.873
null mse: 4.118
"""
