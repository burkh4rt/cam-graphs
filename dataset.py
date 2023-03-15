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
ukb_column_lookup = (
    pd.read_csv(os.path.join("data", "ukb_fields.csv"))[
        ["field.html", "col.name"]
    ]
    .set_index("field.html")
    .to_dict()["col.name"]
)
target_plus_graph_features = [
    "is_fem",
    "age",
    "vol_hyper_25781_2_0",
    "vol_grey_matter_25006_2_0",
    "vol_white_matter_25008_2_0",
    "body_mass_index_bmi_f21001_2_0",
    "mean_time_to_correctly_identify_matches_f20023_2_0",
    "maximum_digits_remembered_correctly_f4282_2_0",
]

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
    .rename(columns=ukb_column_lookup)
    .assign(
        age=lambda df: (
            (
                pd.to_datetime(
                    df["date_of_attending_assessment_centre_f53_2_0"]
                )
                - pd.to_datetime(df["year_of_birth_f34_0_0"], format="%Y")
            )
            / pd.Timedelta("365 days")
        ).round(0),
        is_fem=lambda df: (abs(df["genetic_sex_f22001_0_0"]) < 0.5).astype(
            int
        ),
    )
    .rename(
        columns={
            "total_volume_of_white_matter_hyperintensities_"
            "from_t1_and_t2_flair_images_f25781_2_0": "vol_hyper_25781_2_0",
            "volume_of_grey_matter_f25006_2_0": "vol_grey_matter_25006_2_0",
            "volume_of_white_matter_f25008_2_0": "vol_white_matter_25008_2_0",
        }
    )
    .filter(
        regex="(.*_("
        + "|".join(rois)
        + f").*_vol)|"
        + "|".join(target_plus_graph_features)
    )
    .rename(
        columns=lambda c: "_".join(c.split("_")[1:3]) if "_vol" in c else c
    )
)[target_plus_graph_features + chis_rois]

fids = (
    (~aff1.isna())
    .loc[lambda df: df.all(axis=1)]
    .index.intersection((~s1.isna()).loc[lambda df: df.all(axis=1)].index)
)

train_ids = rng.choice(np.arange(len(fids)), size=10000, replace=False)
val_ids = rng.choice(
    np.setdiff1d(np.arange(len(fids)), train_ids), size=5000, replace=False
)
test_ids = np.setdiff1d(np.arange(len(fids)), np.union1d(train_ids, val_ids))
assert (
    len(np.intersect1d(train_ids, val_ids))
    == len(np.intersect1d(train_ids, test_ids))
    == len(np.intersect1d(val_ids, test_ids))
    == 0
)
assert len(np.union1d(train_ids, np.union1d(val_ids, test_ids))) == len(fids)

aff1 = aff1.loc[fids]
s1 = s1.loc[fids]
y1 = s1[target_plus_graph_features]
s1 = s1.drop(columns=y1.columns)
s1 -= s1.loc[fids[train_ids]].mean(axis=0)
s1 /= s1.loc[fids[train_ids]].std(axis=0)
y1.loc[:, target_plus_graph_features[1:]] -= y1.loc[
    fids[train_ids], target_plus_graph_features[1:]
].mean(axis=0)
y1.loc[:, target_plus_graph_features[1:]] /= y1.loc[
    fids[train_ids], target_plus_graph_features[1:]
].std(axis=0)

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

f_data = lambda f: t_data.Data(
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
        np.array(y1.loc[f, target_plus_graph_features]).reshape(
            -1, len(target_plus_graph_features)
        ),
        dtype=t.float,
    ),
)

num_node_features = f_data(fids[0]).num_node_features
num_nodes = f_data(fids[0]).num_nodes
num_graph_features = len(target_plus_graph_features[1:])

data_list = [f_data(f) for f in fids]
data_train = [data_list[i] for i in train_ids]
data_val = [data_list[i] for i in val_ids]
data_test = [data_list[i] for i in test_ids]

mean_train = np.array(
    y1.loc[fids[train_ids], target_plus_graph_features[0]]
).mean()
std_train = np.array(
    y1.loc[fids[train_ids], target_plus_graph_features[0]]
).std()

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

if __name__ == "__main__":
    print(f"total available: {len(fids)}")
    print(f"training set size: {len(train_ids)}")
    print(f"validation set size: {len(val_ids)}")
    print(f"test set size: {len(test_ids)}")
    print(f"examplar graph:\n {f_data(fids[0])}")


"""
total available: 20113
training set size: 10000
validation set size: 5000
test set size: 5113
examplar graph:
 Data(x=[62, 1], edge_index=[2, 3844], edge_attr=[3844, 1], y=[1, 8])
"""
