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


1. Edit `config.py`. We provide specific description of configuration in 'config.txt' file.

2. Download our [network parameters](https://drive.google.com/drive/folders/1FVpq9VsomdGQU2LF1ryVNSI8q7Y-yVca?usp=sharing) if you want to obtain the performance of the paper.


3. Run with 
```
cd Semantic-Line-SLNet-master
python main.py
```



### Reference
```
@Inproceedings{
Lee2017SLNet,
title={Semantic Line Detection and Its Applications},
author={Jun-Tae Lee, Han-Ul Kim, Chul Lee, and Chang-Su Kim},
booktitle={ICCV},
year={2017}
}
```
