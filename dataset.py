#!/usr/bin/env python3

"""
Wrangles biobank data into a form appropriate for torch geometric
"""

import functools
import os

import pandas as pd
import numpy as np

import torch as t
from torch_geometric.data import Data

rng = np.random.default_rng(0)
target = "20016"

roi_dict = {
    # "amygdala": ["amygdala"],
    # "hippocampus": ["hippocampus"],
    "perirhinal": ["PeEc"],
    "ectorhinal": ["EC"],
    "parahippocampal": ["PHA1", "PHA2", "PHA3"],
    "postcentral_gyrus": ["3a", "3b"],
    "superior_parietal": [
        "7PC",
        "7AL",
        "7Am",
        "7PL",
        "7Pm",
        "VIP",
        "MIP",
        "LIPv",
        "LIPd",
        "AIP",
    ],
    "inferior_parietal": [
        "PFop",
        "PFt",
        "PF",
        "PGs",
        "PGi",
        "PFm",
        "PGp",
        "TPOJ1",
        "TPOJ2",
        "TPOJ3",
        "IP0",
        "IP1",
        "IP2",
        "IPS1",
    ],
}
rois = np.array(functools.reduce(lambda x, y: x + y, roi_dict.values()))
chis = ["L", "R"]
chis_rois = [f"{c}_{r}" for c in chis for r in rois]

senders, receivers = map(
    np.ravel, np.mgrid[0 : len(chis_rois), 0 : len(chis_rois)]
)

aff1 = (
    pd.read_csv(
        os.path.join("data", "affinity_matrices_dementia_only-baseline.csv")
    )
    .set_index("FID")
    .filter(regex="^(?!.*(Hippocampus|Amygdala)).*")
    .rename(
        columns=lambda c: "_".join([d for d in c.split("_") if d != "ROI"])
    )
)

s1 = (
    pd.read_csv(
        os.path.join("data", "biobank-structural_plus-baseline.csv"),
        low_memory=False,
    )
    .set_index("FID")
    .assign(
        age=lambda df: (
            (
                pd.to_datetime(df["53-2.0"])
                - pd.to_datetime(df["34-0.0"], format="%Y")
            )
            / pd.Timedelta("365 days")
        ).round(0),
        is_fem=lambda df: (abs(df["22001-0.0"]) < 0.5).astype(int),
    )
    .filter(
        regex="(.*_(" + "|".join(rois) + f").*_vol)"
        f"|({target}-2.*)|(age)|(is_fem)"
    )
    .rename(
        columns=lambda c: "_".join(c.split("_")[1:3]) if "_vol" in c else c
    )
)[[f"{target}-2.0", "age", "is_fem"] + chis_rois]

fids = (
    (~aff1.isna())
    .loc[lambda df: df.all(axis=1)]
    .index.intersection((~s1.isna()).loc[lambda df: df.all(axis=1)].index)
)

aff1 = aff1.loc[fids]
s1 = s1.loc[fids]

# reconstitute affinity matrices from lower triangular portion
aff1 = pd.concat(
    [
        aff1,
        aff1.rename(columns=lambda c: "_by_".join(c.split("_by_")[::-1])),
        pd.DataFrame(
            index=aff1.index,
            data={
                f"{r}_by_{r}": np.ones(shape=(len(aff1.index),))
                for r in chis_rois
            },
        ),
    ],
    axis=1,
)[
    [
        chis_rois[senders[i]] + "_by_" + chis_rois[receivers[i]]
        for i in range(len(senders))
    ]
]

f_data = lambda f: Data(
    x=t.tensor(
        np.column_stack([s1.loc[f, chis_rois].values]),
        dtype=t.float,
    ),  # node feature matrix of n_nodes x d_node_feat
    edge_index=t.tensor(
        np.row_stack([senders, receivers]), dtype=t.long
    ),  # 2 x n_edges in COO
    edge_attr=t.tensor(
        aff1.loc[f].values.reshape([-1, 1]), dtype=t.float
    ),  # n_edges x d_edge_feat
    y=t.tensor(
        np.array(s1.loc[f, [f"{target}-2.0", "age", "is_fem"]]).reshape(-1, 3),
        dtype=t.float,
    ),
)

num_node_features = f_data(fids[0]).num_node_features
num_nodes = f_data(fids[0]).num_nodes

train_ids = rng.choice(np.arange(len(fids)), size=25000, replace=False)
test_ids = np.setdiff1d(np.arange(len(fids)), train_ids)
assert len(np.intersect1d(train_ids, test_ids)) == 0

data_list = [f_data(f) for f in fids]
data_train = [data_list[i] for i in train_ids]
data_test = [data_list[i] for i in test_ids]

mean_train = np.array(s1.loc[fids[train_ids], f"{target}-2.0"]).mean()
std_train = np.array(s1.loc[fids[train_ids], f"{target}-2.0"]).std()

if __name__ == "__main__":
    print(f"total available: {len(fids)}")
    print(f"training set size: {len(train_ids)}")
    print(f"test set size: {len(test_ids)}")
    print(f"examplar graph:\n {f_data(fids[0])}")


"""
total available: 28375
training set size: 25000
test set size: 3375
examplar graph:
 Data(x=[62, 1], edge_index=[2, 3844], edge_attr=[3844, 1], y=[3, 1])
"""
