# ROGO
This repo contains the codes for our work “[Restricted orthogonal gradient projection for continual learning](https://www.sciencedirect.com/science/article/pii/S2666651023000128)”.


## Data Preparation

The dataset for PMNIST, CIFAR-100, and Mixture will be automatically downloaded. For the experiments on MiniImageNet, please download the [train_data](https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view) and [test_data](https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view), and place them under the `data` folder.

## Experiments

This repository currently contains experiments reported in the paper for Permuted MNIST, 10-split CIFAR-100, 20-split MiniImageNet and the Mixture datasets. All these experiments can be run using the following command:

```
bash run.sh
```

## Citation

```
@article{YANG202398,
title = {Restricted orthogonal gradient projection for continual learning},
journal = {AI Open},
volume = {4},
pages = {98-110},
year = {2023},
issn = {2666-6510},
doi = {https://doi.org/10.1016/j.aiopen.2023.08.010},
url = {https://www.sciencedirect.com/science/article/pii/S2666651023000128},
author = {Zeyuan Yang and Zonghan Yang and Yichen Liu and Peng Li and Yang Liu},
}
```