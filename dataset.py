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
aff2 = (
    pd.read_csv(
        os.path.join("data", "affinity_matrices_dementia_only-follow_up.csv")
    )
    .set_index("FID")
    .filter(regex="^(?!.*(Hippocampus|Amygdala)).*")
    .rename(
        columns=lambda c: "_".join([d for d in c.split("_") if d != "ROI"])
    )
)

s1 = (
    pd.read_csv(
        os.path.join("data", "structural_plus-baseline.csv"),
        low_memory=False,
    )
    .set_index("FID")
    .filter(regex="(.*_(" + "|".join(rois) + ").*_vol)|(Vol_total_volume)")
    .rename(
        columns=lambda c: "_".join(c.split("_")[1:3])
        if "total" not in c
        else c
    )
)[["Vol_total_volume"] + chis_rois]

s2 = (
    pd.read_csv(
        os.path.join("data", "structural_plus-follow_up.csv"),
        low_memory=False,
    )
    .set_index("FID")
    .filter(regex="(.*[-_](" + "|".join(rois) + ").*_vol.*)")
    .rename(
        columns=lambda c: "_".join(c.split("_")[:2]) if "total" not in c else c
    )
)[chis_rois]

fids = (
    (~aff1.isna())
    .loc[lambda df: df.all(axis=1)]
    .index.intersection((~aff2.isna()).loc[lambda df: df.all(axis=1)].index)
    .intersection((~s1.isna()).loc[lambda df: df.all(axis=1)].index)
    .intersection((~s2.isna()).loc[lambda df: df.all(axis=1)].index)
    .intersection(s1.loc[lambda df: df["Vol_total_volume"] != -1.0].index)
)

aff1 = aff1.loc[fids]
aff2 = aff2.loc[fids]
s1 = s1.loc[fids]
s2 = s2.loc[fids]

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
        np.column_stack(
            [s1.loc[f, chis_rois].values, s2.loc[f, chis_rois].values]
        ),
        dtype=t.float,
    ),  # node feature matrix of n_nodes x d_node_feat
    edge_index=t.tensor(
        np.row_stack([senders, receivers]), dtype=t.long
    ),  # 2 x n_edges COO
    edge_attr=t.tensor(
        aff1.loc[f].values.reshape([-1, 1]), dtype=t.float
    ),  # n_edges x d_edge_feat
    y=t.tensor(
        np.array(s1.loc[f, "Vol_total_volume"]).reshape(1, -1),
        dtype=t.float,
    ),
)

num_node_features = f_data(fids[0]).num_node_features
num_nodes = f_data(fids[0]).num_nodes


train_ids = rng.choice(np.arange(len(fids)), size=800, replace=False)
test_ids = np.setdiff1d(np.arange(len(fids)), train_ids)
assert len(np.intersect1d(train_ids, test_ids)) == 0

data_list = [f_data(f) for f in fids]
data_train = [data_list[i] for i in train_ids]
data_test = [data_list[i] for i in test_ids]

mean_train = np.array(s1.loc[fids[train_ids], "Vol_total_volume"]).mean()
