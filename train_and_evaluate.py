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
  0%|          | 0/10 [00:00<?, ?it/s]61.468
 10%|█         | 1/10 [00:11<01:40, 11.16s/it]42.386
 20%|██        | 2/10 [00:22<01:27, 10.98s/it]39.117
 30%|███       | 3/10 [00:32<01:16, 10.86s/it]38.024
 40%|████      | 4/10 [00:43<01:04, 10.79s/it]37.130
 50%|█████     | 5/10 [00:54<00:54, 10.81s/it]36.803
 60%|██████    | 6/10 [01:05<00:43, 10.95s/it]36.848
 70%|███████   | 7/10 [01:16<00:32, 10.89s/it]36.785
 80%|████████  | 8/10 [01:27<00:22, 11.04s/it]36.184
 90%|█████████ | 9/10 [01:38<00:11, 11.10s/it]36.752
100%|██████████| 10/10 [01:49<00:00, 11.00s/it]
val mse:  35.707
test mse: 35.733
test mse (null): 54.910
executed in 111.69 seconds
"""
