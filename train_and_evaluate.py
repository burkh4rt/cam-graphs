#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import datetime
import os

import numpy as np

import torch as t
import torch_geometric.loader as t_loader
import torch_geometric.explain as t_explain
import tqdm

import dataset
import model

t.manual_seed(0)


def evaluate(
    mdl: model.GCN, return_preds=False
) -> float | tuple[float, dict[str, np.array]]:
    mdl.eval()
    loader_test = t_loader.DataLoader(
        dataset.data_test, batch_size=len(dataset.test_ids)
    )
    batch_test = next(iter(loader_test))
    preds_test = mdl(
        batch_test.x,
        batch_test.edge_index,
        batch_test.edge_attr,
        batch_test.batch,
        batch_test.y[:, 1:],
    )
    ypred = preds_test.detach().numpy().ravel()
    ytest = batch_test.y[:, 0].numpy().ravel()
    mse_test = np.mean(np.square(ypred - ytest))
    mse_null = np.mean(np.square(dataset.mean_train - ytest))
    mdl.train()
    if return_preds:
        return mse_test / mse_null, {"preds": ypred, "trues": ytest}
    else:
        return mse_test / mse_null


def train(mdl: model.GCN) -> model.GCN:
    mdl.train()
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = t.nn.MSELoss()

    for _ in tqdm.tqdm(range(10)):
        loader_train = t_loader.DataLoader(
            dataset.data_train, 1000  # batch_size=len(dataset.train_ids)
        )
        for data in iter(loader_train):
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
    return mdl


if __name__ == "__main__":
    mdl = model.GCN()
    mdl = train(mdl)
    t.save(
        mdl.state_dict(),
        os.path.join(
            "tmp",
            "mdl-fluid_intell-"
            + datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%MZ"
            )
            + ".ckpt",
        ),
    )

    n_mse, pred_true_dict = evaluate(mdl, return_preds=True)
    print(f"norm. mse: {n_mse:.2f}")

    ytrue, ypred = pred_true_dict["trues"], pred_true_dict["preds"]

    print(
        "rmse null: {rmse:.2f}".format(
            rmse=np.sqrt(np.mean(np.square(ytrue - dataset.mean_train)))
        )
    )
    print(
        "rmse ours: {rmse:.2f}".format(
            rmse=np.sqrt(np.mean(np.square(ytrue - ypred)))
        )
    )

    explainer = t_explain.Explainer(
        model=mdl,
        algorithm=t_explain.GNNExplainer(epochs=20),
        explainer_config=t_explain.ExplainerConfig(
            explanation_type="model", node_mask_type="attributes", edge_mask_type='object',
        ),
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
        threshold_config=dict(threshold_type="topk", value=10),
    )

    explanation = explainer(
        dataset.batch_test.x,
        dataset.batch_test.edge_index,
        edge_attr=dataset.batch_test.edge_attr,
        batch=dataset.batch_test.batch,
        graph_feats=dataset.batch_test.y[:, 1:],
    )

    # print(f"Generated explanations in {explanation.available_explanations}")
    # explanation.visualize_feature_importance(
    #     os.path.join(
    #         "figures",
    #         "-".join(
    #             [
    #                 "feature-importances",
    #                 "mdl-fluid_intell-"
    #                 + datetime.datetime.now(datetime.timezone.utc).strftime(
    #                     "%Y%m%dT%H%MZ"
    #                 ) + ".pdf",
    #             ]
    #         ),
    #     ),
    #     top_k=10,
    # )


""" output:
norm. mse: 0.86
rmse null: 2.04
rmse ours: 1.90
executed in 216.13 seconds
"""
