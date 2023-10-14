# gnns-graphs

This code trains a simple graph neural network on bacs/biobank data using
[pytorch-geometric](https://pytorch-geometric.readthedocs.io). Hyperparameters
are tuned with [optuna](https://optuna.org). Feature importances can then be
assigned with [shap](https://shap.readthedocs.io).

The code operates on datasets that are proprietary, and so is only shared for
illustrative purposes. The main script is [xvalidate.py](./xvalidate.py).

Feel free to reuse/repurpose as you see fit.

<!---
format code with:
```
black .
prettier --write --print-width 79 --prose-wrap always *.md
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
