![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [ICCV 2017] Semantic Line Detection and Its Applications

<!--
![IVOS Image](Overall_Network.png)

\\[[Project page]](https://openreview.net/forum?id=bo_lWt_aA)
\\[[arXiv]](https://arxiv.org/abs/2007.08139)
-->

Pytorch reimplementation for the paper **"Semantic Line Detection and Its Applications"**.

### Requirements
- PyTorch 1.3.1
- CUDA 10.0
- CuDNN 7.6.5
- python 3.6

### Installation
Create conda environment:
```
    $ conda create -n SLNet python=3.6 anaconda
    $ conda activate SLNet
    $ pip install opencv-python==3.4.2.16
    $ conda install pytorch==1.3.1 torchvision cudatoolkit=10.0 -c pytorch
```

Download repository:
```
    $ git clone https://github.com/dongkwonjin/Semantic-Line-SLNet.git
```
### Instruction

Our model checpoints are available at [here](https://drive.google.com/drive/folders/1QJDfjSo8blL4KgG8bid5N34GicHUhHKF?usp=sharing)



### Reference
```
@Inproceedings{
Lee2017SLNet,
title={Semantic Line Detection and Its Applications},
author={Jun-Tae Lee, Han-Ul Kim, Chul Lee, and Chang-Su Kim},
booktitle={ICCV},
year={2017},

}
```
