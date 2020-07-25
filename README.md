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

1. Download [SEL dataset and preprocessed data](https://drive.google.com/file/d/1jk43D2wHdF84TfoAUbWjZWenVYnpcD4o/view?usp=sharing). You can download the original dataset in [here](http://mcl.korea.ac.kr/research/Submitted/jtlee_slnet/ICCV2017_JTLEE_dataset.7z). We provide the preprocessed data to train and test proposed three networks in ```data``` and ```edge``` folder. We obtain the data in  ```edge``` folder, by employing [HED algorithm]

2. Edit `config.py`. Please modify ```dataset_dir```. We provide specific description of configuration in 'config.txt' file.

3. Download our [network parameters](https://drive.google.com/file/d/1jrcu3R90U9XeG-jpOWIcoXcHimPsIaV-/view?usp=sharing) if you want to get the performance of the paper.


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
