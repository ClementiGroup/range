![test](banner.png)

RANGE is a model-agnostic framework that employs an attention-based aggregation-broadcast mechanism to significantly reduce oversquashing effects in message-passing neural networks.

## Prerequisites
Install the following prerequisites:
```bash
conda install python==3.12
pip install --extra-index-url=https://download.pytorch.org/whl/cu124 torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install lightning tensorboard torchtnt
```

## Installation
Install the lightning2 branch of the mlcg package:
```bash
git clone -b lightning2 --single-branch git@github.com:ClementiGroup/mlcg.git
cd mlcg; pip install .; cd ..
```

Install the range package
```bash
git clone git@github.com:ClementiGroup/range.git
cd range; pip install .; cd ..
```
