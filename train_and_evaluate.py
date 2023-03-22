#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import sklearn.metrics as skl_mets
import torch as t
import torch_geometric.loader as t_loader
import tqdm as tqdm

import dataset
import model

t.manual_seed(0)
criterion = t.nn.functional.binary_cross_entropy


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


def train(mdl: model.GCN) -> model.GCN:
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in tqdm.tqdm(range(10)):
        print("{:.3f}".format(loss_val(mdl).detach().numpy()))
        mdl.train()
        loader_train = t_loader.DataLoader(dataset.data_train, 1000)
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
    import time

    t0 = time.time()

    mdl = model.GCN()
    mdl = train(mdl)
    mdl.eval()
    ypred = (
        mdl(
            dataset.batch_test.x,
            dataset.batch_test.edge_index,
            dataset.batch_test.edge_attr,
            dataset.batch_test.batch,
            dataset.batch_test.y[:, 1:],
        )
        .detach()
        .numpy()
        .ravel()
    )

    ytrue = dataset.batch_test.y[:, 0].numpy().ravel()

    print("auc: {auc:.2f}".format(auc=skl_mets.roc_auc_score(ytrue, ypred)))
    print(
        "acc: {acc:.2f}".format(
            acc=skl_mets.accuracy_score(ytrue, (ypred > 0.5).astype(int))
        )
    )
    print(
        "f1:  {f1:.2f}".format(
            f1=skl_mets.f1_score(ytrue, (ypred > 0.5).astype(int))
        )
    )

    # explainer = t_explain.Explainer(
    #     model=mdl,
    #     algorithm=t_explain.GNNExplainer(epochs=20),
    #     explainer_config=t_explain.ExplainerConfig(
    #         explanation_type="model",
    #         node_mask_type="attributes",
    #         edge_mask_type="object",
    #     ),
    #     model_config=dict(
    #         mode="regression",
    #         task_level="graph",
    #         return_type="raw",
    #     ),
    #     threshold_config=dict(threshold_type="topk", value=10),
    # )

    # explanation = explainer(
    #     dataset.batch_test.x,
    #     dataset.batch_test.edge_index,
    #     edge_attr=dataset.batch_test.edge_attr,
    #     batch=dataset.batch_test.batch,
    #     graph_feats=dataset.batch_test.y[:, 1:],
    # )

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
  0%|          | 0/10 [00:00<?, ?it/s]0.695
 10%|█         | 1/10 [00:15<02:17, 15.25s/it]0.505
 20%|██        | 2/10 [00:30<02:01, 15.16s/it]0.457
 30%|███       | 3/10 [00:45<01:46, 15.16s/it]0.437
 40%|████      | 4/10 [01:00<01:30, 15.04s/it]0.427
 50%|█████     | 5/10 [01:15<01:15, 15.01s/it]0.422
 60%|██████    | 6/10 [01:30<00:59, 15.00s/it]0.424
 70%|███████   | 7/10 [01:45<00:45, 15.10s/it]0.419
 80%|████████  | 8/10 [02:01<00:30, 15.32s/it]0.416
 90%|█████████ | 9/10 [02:17<00:15, 15.45s/it]0.418
100%|██████████| 10/10 [02:32<00:00, 15.27s/it]
auc: 0.88
acc: 0.79
f1:  0.81
"""
