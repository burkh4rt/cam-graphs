#!/usr/bin/env python3

"""
Wrangles biobank data into a form appropriate for torch geometric
"""

import functools
import os

import pandas as pd
import numpy as np

import torch as t
import torch_geometric.loader as t_loader
import torch_geometric.data as t_data

rng = np.random.default_rng(0)

col_target = "av1451_age"

df_bacs = (
    pd.read_csv(os.path.join("data", "bacs_structural_data.csv"))
    .assign(
        id_dt=lambda df: df.scan_category_subject_id
        + "_"
        + df.av1451_av1451_date.str.split(" ").str[0],
        apoe4pos=lambda df: (
            (df.scan_category_apoe1 == "E4") | (df.scan_category_apoe2 == "E4")
        ).astype(int),
        is_female=lambda df: df.scan_category_sex.isin(["Female"]).astype(int),
    )
    .drop(
        columns=[
            "scan_category_apoe1",
            "scan_category_apoe2",
            "scan_category_sex",
        ]
    )
    .set_index("id_dt")
)

rois = [
    "_".join(x.split("_")[1:])
    for x in df_bacs.filter(regex="av1451_[0000-9999]").columns
]

# the mri and av1451 columns are just f"mri_{roi}" and f"av1451_{roi}"
assert set(df_bacs.filter(regex="mri_[0000-9999]").columns) == set(
    f"mri_{r}" for r in rois
)
assert set(df_bacs.filter(regex="av1451_[0000-9999]").columns) == set(
    f"av1451_{r}" for r in rois
)

cols_feats = [f"{mo}_{r}" for r in rois for mo in ["mri", "av1451"]] + [
    "apoe4pos",
    "is_female",
]

ids = (
    df_bacs[cols_feats + [col_target]]
    .loc[lambda df: ~df.isna().any(axis=1)]
    .index.to_list()
)

n_ids = len(ids)
train_ids = rng.choice(ids, size=int(0.65 * n_ids), replace=False)
val_ids = rng.choice(
    np.setdiff1d(ids, train_ids), size=int(0.15 * n_ids), replace=False
)
test_ids = np.setdiff1d(ids, np.union1d(train_ids, val_ids))

# ids are partitioned into train, val, and test
assert (
    len(np.intersect1d(train_ids, val_ids))
    == len(np.intersect1d(train_ids, test_ids))
    == len(np.intersect1d(val_ids, test_ids))
    == 0
)
assert len(np.union1d(train_ids, np.union1d(val_ids, test_ids))) == len(ids)

# normalise features according to training set
df_bacs.loc[:, cols_feats] -= df_bacs.loc[train_ids, cols_feats].mean(axis=0)
df_bacs.loc[:, cols_feats] /= df_bacs.loc[train_ids, cols_feats].std(axis=0)

mean_train = df_bacs.loc[train_ids, col_target].mean()
std_train = df_bacs.loc[train_ids, col_target].std()

# grab correlation matrix between rois
c_mat_bacs = (
    pd.read_csv(os.path.join("data", "bacs_correlations.csv"), index_col=0)
    .rename(index=lambda x: "_".join(x.split("_")[1:-1]))
    .rename(columns=lambda x: "_".join(x.split("_")[1:-1]))
)
assert set(c_mat_bacs.columns) == set(c_mat_bacs.index) == set(rois)

senders, receivers = map(np.ravel, np.mgrid[0 : len(rois), 0 : len(rois)])

f_data = lambda f: t_data.Data(
    x=t.tensor(
        np.column_stack(
            [
                df_bacs.loc[f, [f"{mo}_{r}" for r in rois]].values
                for mo in ["mri", "av1451"]
            ]
        ).astype(float),
        dtype=t.float,
    ),  # node feature matrix of n_nodes x d_node_feat
    edge_index=t.tensor(
        np.row_stack([senders, receivers]), dtype=t.long
    ),  # 2 x n_edges in COO
    edge_attr=t.tensor(
        c_mat_bacs.values[senders, receivers], dtype=t.float
    ),  # n_edges x d_edge_feat
    y=t.tensor(
        df_bacs.loc[f, [col_target, "apoe4pos", "is_female"]]
        .values.astype(float)
        .reshape(1, -1),
        dtype=t.float,
    ),  # 1 x n_graph_feats
)

num_node_features = f_data(ids[0]).num_node_features
num_nodes = f_data(ids[0]).num_nodes
num_graph_features = len(f_data(ids[0]).y.ravel()) - 1

data_train = [f_data(i) for i in train_ids]
data_val = [f_data(i) for i in val_ids]
data_test = [f_data(i) for i in test_ids]

batch_val = next(
    iter(
        t_loader.DataLoader(
            data_val,
            batch_size=len(val_ids),
            shuffle=False,
        )
    )
)

batch_test = next(
    iter(
        t_loader.DataLoader(
            data_test,
            batch_size=len(test_ids),
            shuffle=False,
        )
    )
)

batch_0 = next(
    iter(
        t_loader.DataLoader(
            data_test,
            batch_size=1,
            shuffle=False,
        )
    )
)

cols_xy1_ravelled = np.concatenate(
    [
        np.column_stack(
            [[f"{mo}_{r}" for r in rois] for mo in ["mri", "av1451"]]
        ).reshape(-1),
        np.array(["apoe4pos", "is_female"]),
    ]
)


if __name__ == "__main__":
    print(f"total available: {len(ids)}")
    print(f"training set size: {len(train_ids)}")
    print(f"validation set size: {len(val_ids)}")
    print(f"test set size: {len(test_ids)}")
    print(f"examplar graph:\n {f_data(ids[0])}")

"""
total available: 222
training set size: 144
validation set size: 33
test set size: 45
examplar graph:
 Data(x=[113, 2], edge_index=[2, 12769], edge_attr=[12769], y=[1, 3])
"""
