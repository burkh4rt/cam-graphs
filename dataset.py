#!/usr/bin/env python3

"""
Wrangles BACS data into a form appropriate for torch geometric

The main class `dataset` forms cross-validated datasets for you and can be
initialised with a fold number 0<=`fold`<n_folds for convenience
"""
import itertools
import os

import pandas as pd
import numpy as np

import torch as t
import torch_geometric.loader as t_loader
import torch_geometric.data as t_data

rng = np.random.default_rng(0)

# set number of folds for cross-validation
n_folds = 6

# predictive target for supervised modelling
col_target = "av1451_age"

# modalities -- can also include "mri" in the following list
mod_list = ["av1451", "mri"]

# load everything from bacs
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

# regions of interest for which we have tau-PET data
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

# features to be used by the model
cols_feats = [f"{mo}_{r}" for r in rois for mo in mod_list] + [
    "apoe4pos",
    "is_female",
]

# persons with full data
ids = (
    df_bacs[cols_feats + [col_target]]
    .loc[lambda df: ~df.isna().any(axis=1)]
    .index.to_numpy()
)

# grab correlation matrix between rois
c_mat_bacs = (
    pd.read_csv(os.path.join("data", "bacs_correlations.csv"), index_col=0)
    .rename(index=lambda x: "_".join(x.split("_")[1:-1]))
    .rename(columns=lambda x: "_".join(x.split("_")[1:-1]))
)
assert set(c_mat_bacs.columns) == set(c_mat_bacs.index) == set(rois)

# coordinates to express edges in sparse COOrdinate format
senders, receivers = map(np.ravel, np.mgrid[0 : len(rois), 0 : len(rois)])

# returns a graph for the person with id=f
f_data = lambda f, df_bacs: t_data.Data(
    x=t.tensor(
        np.column_stack(
            [
                df_bacs.loc[f, [f"{mo}_{r}" for r in rois]].values
                for mo in mod_list
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

n_ids = len(ids)

# make sure all visits from the same subject lie in the same fold
subjects, s_inv = np.unique(
    [x.split("_")[0] for x in ids], return_inverse=True
)
assert np.all(subjects[s_inv] == np.array([x.split("_")[0] for x in ids]))
subjects_folds = rng.choice(n_folds, size=n_ids)
folds = subjects_folds[s_inv]


class dataset:
    """returns a dataset corresponding to testing on persons in fold `fold`,
    validating on persons in fold `fold`+1, and training on persons from all
    other folds"""

    def __init__(self, fold: int):
        assert 0 <= fold < n_folds
        self.ids = ids
        self.n_ids = n_ids
        self.fold = fold
        self.folds = folds
        self.test_ids = self.ids[self.folds == self.fold]
        self.val_ids = self.ids[self.folds == (self.fold + 1) % n_folds]
        self.train_ids = np.setdiff1d(
            self.ids, np.union1d(self.test_ids, self.val_ids)
        )

        # the training, validation, & test sets partition the data
        for p1, p2 in itertools.combinations(
            [self.train_ids, self.val_ids, self.test_ids], 2
        ):
            assert len(np.intersect1d(p1, p2)) == 0
        assert len(self.train_ids) + len(self.val_ids) + len(
            self.test_ids
        ) == len(self.ids)

        self.df_bacs = df_bacs.copy(deep=True)
        self.c_mat_bacs = c_mat_bacs

        # normalise features according to training set
        self.df_bacs.loc[:, cols_feats] -= df_bacs.loc[
            self.train_ids, cols_feats
        ].mean(axis=0)
        self.df_bacs.loc[:, cols_feats] /= df_bacs.loc[
            self.train_ids, cols_feats
        ].std(axis=0)

        self.mean_train = df_bacs.loc[self.train_ids, col_target].mean()
        self.std_train = df_bacs.loc[self.train_ids, col_target].std()

        self.f_data = lambda f: f_data(f, self.df_bacs)

        self.num_node_features = self.f_data(ids[0]).num_node_features
        self.num_nodes = self.f_data(ids[0]).num_nodes
        self.num_graph_features = len(self.f_data(ids[0]).y.ravel()) - 1

        self.data_train = [self.f_data(i) for i in self.train_ids]
        self.data_val = [self.f_data(i) for i in self.val_ids]
        self.data_test = [self.f_data(i) for i in self.test_ids]

        self.batch_val = next(
            iter(
                t_loader.DataLoader(
                    self.data_val,
                    batch_size=len(self.val_ids),
                    shuffle=False,
                )
            )
        )

        self.batch_test = next(
            iter(
                t_loader.DataLoader(
                    self.data_test,
                    batch_size=len(self.test_ids),
                    shuffle=False,
                )
            )
        )
        self.batch_0 = next(
            iter(
                t_loader.DataLoader(
                    self.data_test,
                    batch_size=1,
                    shuffle=False,
                )
            )
        )

        self.cols_xy1_ravelled = np.concatenate(
            [
                np.column_stack(
                    [[f"{mo}_{r}" for r in rois] for mo in mod_list]
                ).reshape(-1),
                np.array(["apoe4pos", "is_female"]),
            ]
        )


if __name__ == "__main__":
    dset = dataset(0)
    print(f"total available: {len(dset.ids)}")
    print(f"training set size: {len(dset.train_ids)}")
    print(f"validation set size: {len(dset.val_ids)}")
    print(f"test set size: {len(dset.test_ids)}")
    print(f"examplar graph:\n {dset.f_data(ids[0])}")

    # check that we're cross-validating
    for f1, f2 in itertools.combinations(range(n_folds), 2):
        assert (
            len(np.intersect1d(dataset(f1).test_ids, dataset(f2).test_ids))
            == 0
        )

"""
total available: 222
training set size: 156
validation set size: 29
test set size: 37
examplar graph:
 Data(x=[113, 2], edge_index=[2, 12769], edge_attr=[12769], y=[1, 3])
"""
