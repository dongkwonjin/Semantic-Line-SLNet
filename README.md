![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [ICCV 2017] Semantic Line Detection and Its Applications

<!--
![IVOS Image](Overall_Network.png)

\\[[Project page]](https://openreview.net/forum?id=bo_lWt_aA)
\\[[arXiv]](https://arxiv.org/abs/2007.08139)
-->

Official pytorch reimplementation for **"Semantic Line Detection and Its Applications"** [[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Semantic_Line_Detection_ICCV_2017_paper.pdf)

Recent works can be found in [here](https://github.com/dongkwonjin/Semantic-Line-DRM) and [here](https://github.com/dongkwonjin/Semantic-Line-MWCS).

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

1. Download [SEL dataset and preprocessed data](https://drive.google.com/file/d/1Urfbj7IvfDpMw7b9SAgOF3Da6hTWrPhL/view?usp=sharing) to ```root/```. You can download the original dataset in [here](http://mcl.korea.ac.kr/jtlee_iccv2017/). We provide the preprocessed data to train and test the model in ```data``` and ```edge``` folder. You can generate these data using the source codes in Preprocessing/. We obtain the data in  ```edge``` folder, by employing [HED algorithm](https://github.com/sniklaus/pytorch-hed).

2. Download our [network parameters](https://drive.google.com/file/d/1v2iy8w4FWB7xn_bpwQqR6cVKTftAcQ3o/view?usp=sharing) to ```root/``` if you want to get the performance of the paper.

3. Edit `config.py`. Please modify ```dataset_dir``` and ```paper_weight_dir```.

4. Run with 
```
$ cd Semantic-Line-SLNet-master/(Modeling or Preprocessing)/code/
$ python main.py
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
