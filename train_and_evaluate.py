#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import torch as t
from torch_geometric.loader import DataLoader
import tqdm

import dataset
import model

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


def loss_test_null():
    return criterion(
        dataset.mean_train
        * t.ones_like(dataset.batch_test.y[:, 0].reshape(-1, 1)),
        dataset.batch_test.y[:, 0].reshape(-1, 1),
    )


def train(mdl: model.GCN) -> model.GCN:
    optimizer = t.optim.Adagrad(mdl.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in tqdm.tqdm(range(10)):
        print("{:.3f}".format(loss_val(mdl).detach().numpy()))
        mdl.train()
        loader_train = DataLoader(
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
    import time

    # print contents of file to screen to remember what settings used when
    # running multiple versions simultaneously
    # with open(__file__) as f:
    #     print(f.read())
    #     print("%" * 79)
    # with open(model.__file__) as f:
    #     print(f.read())
    #     print("%" * 79)

    t0 = time.time()

    mdl = model.GCN()
    mdl = train(mdl)

    print("val mse:  {:.3f}".format(loss_val(mdl).detach().numpy()))
    print("test mse: {:.3f}".format(loss_test(mdl).detach().numpy()))
    print("test mse (null): {:.3f}".format(loss_test_null().detach().numpy()))

    print(f"executed in {time.time()-t0:.2f} seconds")

""" output:
  0%|          | 0/10 [00:00<?, ?it/s]61.633
 10%|█         | 1/10 [00:17<02:39, 17.67s/it]42.967
 20%|██        | 2/10 [00:37<02:30, 18.83s/it]37.162
 30%|███       | 3/10 [00:56<02:13, 19.08s/it]35.028
 40%|████      | 4/10 [01:15<01:53, 18.90s/it]33.820
 50%|█████     | 5/10 [01:33<01:33, 18.62s/it]33.412
 60%|██████    | 6/10 [01:50<01:12, 18.23s/it]33.012
 70%|███████   | 7/10 [02:09<00:54, 18.28s/it]32.887
 80%|████████  | 8/10 [02:27<00:36, 18.25s/it]32.666
 90%|█████████ | 9/10 [02:45<00:18, 18.21s/it]32.550
100%|██████████| 10/10 [03:04<00:00, 18.44s/it]
val mse:  32.279
test mse: 32.069
test mse (null): 54.910
executed in 186.16 seconds
"""
