# cam-ukbb-graphs

This code trains a simple graph neural network on biobank data using
[pytorch-geometric](https://pytorch-geometric.readthedocs.io). Hyperparameters
are tuned with the [optuna](https://optuna.org) package.

It requires datasets available on zfs to be placed in the folder `data`:
`/zfs/mcb93/ukbb-graphs/data` and an environment as described in the
`requirements.txt` file.

<!---
format code with:
```
black .
prettier --write --print-width 79 --prose-wrap always *.md
```

Send all to zfs with:
```sh
rsync -avhtXE \
    --chmod=770 \
    --delete \
    --force \
    --groupmap="*:abg" \
    ~/Documents/cambridge/ukbb-graphs \
    abg-cluster1.psychol.private.cam.ac.uk:/zfs/mcb93/ukbb-graphs
```

```
source flashlight/bin/activate
pip3 list --format=freeze > requirements.txt
```
-->
