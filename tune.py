#!/usr/bin/env python3

"""
Imports dataset and model, runs training, and evaluates on a held-out dataset
"""

import datetime
import os

import torch as t
from torch_geometric.loader import DataLoader

import ray.tune as tune
import ray.air as air

import dataset
import model

t.manual_seed(0)


def train(config: dict):
    mdl = model.GCN()
    optimizer = t.optim.Adagrad(
        mdl.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    criterion = t.nn.MSELoss()

    for _ in range(10):
        loader_train = DataLoader(
            dataset.data_train, batch_size=1000, shuffle=True
        )
        for data in iter(loader_train):
            mdl.train()
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

            mdl.eval()
            batch_val = next(
                iter(
                    DataLoader(
                        dataset.data_val,
                        batch_size=len(dataset.val_ids),
                        shuffle=False,
                    )
                )
            )
            loss_val = criterion(
                mdl(
                    batch_val.x,
                    batch_val.edge_index,
                    batch_val.edge_attr,
                    batch_val.batch,
                    batch_val.y[:, 1:],
                ),
                batch_val.y[:, 0],
            )
            print(loss_val)
            tune.report(mse=loss_val)
        t.save(
            mdl.state_dict(),
            os.path.join(
                os.path.dirname(__file__),
                "tmp",
                "mdl-fluid_intell-"
                + datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y%m%dT%H%MZ"
                )
                + ".ckpt",
            ),
        )
    return mdl


if __name__ == "__main__":
    tuner = tune.Tuner(
        train,
        param_space={
            "lr": tune.loguniform(1e-3, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-3),
        },
        tune_config=tune.TuneConfig(mode="min", metric="mse"),
        run_config=air.RunConfig(
            local_dir=os.path.join(os.path.dirname(__file__), "tmp-tuner"),
            name="tuner_experiment",
        ),
    )
    results = tuner.fit()

""" output:
...
(train pid=13829) tensor(4.5546, grad_fn=<MseLossBackward0>)
== Status ==
Current time: 2023-03-14 09:37:01 (running for 00:07:00.16)
Memory usage on this node: 11.6/16.0 GiB
Using FIFO scheduling algorithm.
Resources requested: 1.0/8 CPUs, 0/0 GPUs, 0.0/6.78 GiB heap, 0.0/2.0 GiB objects
Current best trial: cb94e_00000 with mse=4.554620742797852 and parameters={'lr': 0.0022279173681847534, 'wd': 0.0002175001235384768}
Result logdir: /Users/michael/Documents/cambridge/ukbb-graphs/tmp-tuner/tuner_experiment
Number of trials: 1/1 (1 RUNNING)
+-------------------+----------+-----------------+------------+-----------+--------+------------------+
| Trial name        | status   | loc             |         lr |        wd |   iter |   total time (s) |
|-------------------+----------+-----------------+------------+-----------+--------+------------------|
| train_cb94e_00000 | RUNNING  | 127.0.0.1:13829 | 0.00222792 | 0.0002175 |    100 |          402.895 |
+-------------------+----------+-----------------+------------+-----------+--------+------------------+
Result for train_cb94e_00000:
  date: 2023-03-14_09-37-00
  done: false
  experiment_id: 389e5db4badb41e1bee31e4a2a530067
  experiment_tag: 0_lr=0.0022,wd=0.0002
  hostname: dhcp-10-249-24-137.eduroam.wireless.private.cam.ac.uk
  iterations_since_restore: 100
  mse: tensor(4.5546, requires_grad=True)
  node_ip: 127.0.0.1
  pid: 13829
  time_since_restore: 402.89483094215393
  time_this_iter_s: 3.906912088394165
  time_total_s: 402.89483094215393
  timestamp: 1678786620
  timesteps_since_restore: 0
  training_iteration: 100
  trial_id: cb94e_00000
  warmup_time: 0.0022509098052978516
...
"""
