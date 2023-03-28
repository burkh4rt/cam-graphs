# gnns-graphs

This code trains a simple graph neural network on bacs/biobank data using
[pytorch-geometric](https://pytorch-geometric.readthedocs.io). Hyperparameters
are tuned with [optuna](https://optuna.org). Feature importances can then be
assigned with [shap](https://shap.readthedocs.io).

It requires datasets available on zfs to be placed in the folder `data`:
`/zfs/mcb93/gnns-graphs/data` and an environment as described in the
`requirements.txt` file.

The code can also be run with [docker](https://www.docker.com):
```sh
docker build -t burkh4rt/pyg .
docker run --rm -ti -v $(pwd):/home/felixity burkh4rt/pyg model.py
```

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
    ~/Documents/cambridge/gnns-graphs/ \
    abg-cluster1.psychol.private.cam.ac.uk:/zfs/mcb93/gnns-graphs
```

Create venv:
```
python3 -m venv flashlight
source flashlight/bin/activate
pip3 install torch torchvision torchaudio
pip3 install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip3 install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv
pip3 install shap optuna matplotlib
```

Create `reqirements.txt` file:
```
source flashlight/bin/activate
python3 -m pip list --format=freeze > requirements.txt
```

```
docker buildx create --use --name mybuild node-amd64
docker buildx create --append --name mybuild node-arm64
docker buildx build --platform linux/arm64,linux/amd64 -t burkh4rt/pyg:latest --push .
```
-->
