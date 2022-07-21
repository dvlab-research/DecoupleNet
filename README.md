# DecoupleNet
Official implementation for our ECCV 2022 paper "DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation" [[arXiv](https://arxiv.org/pdf/2207.09988.pdf)]

<div align="center">
  <img src="figs/fig.pdf"/>
</div>

# Get Started

## Datasets Preparation

### GTA5

### Cityscapes

## Training

### GTA5 -> Cityspcaes
First, download the pretrained ResNet101 (PyTorch) and sourceonly model from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154502_link_cuhk_edu_hk/EVowKrywcUVJhK0tbO_ebxQBv83FCISbGW_2fTeCWiFvGA), and put them into the directory `./pretrained`.
```
mkdir pretrained && cd pretrained
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
# Also put the sourceonly.pth into ./pretrained/
```

First-phase training:
```
python3 train_phase1.py --snapshot-dir ./snapshots/phase1_reproduce --batch-size 8 --gpus 0,1,2,3 --dist --tensorboard --batch_size_val 4 --src_rootpath [YOUR_SOURCE_DATA_ROOT] --tgt_rootpath [YOUR_TARGET_DATA_ROOT]
```

Second-phase training:
Comming soon

# Acknowledgement
This repository borrows codes from the following repos. Many thanks to the authors for their great work.

ProDA: https://github.com/microsoft/ProDA

FADA: https://github.com/JDAI-CV/FADA

semseg: https://github.com/hszhao/semseg

# Citation
If you find this project useful, please consider citing:

```
@inproceedings{lai2022decouplenet,
  title     = {DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation},
  author    = {Xin Lai, Zhuotao Tian, Xiaogang Xu, Yingcong Chen, Shu Liu, Hengshuang Zhao, Liwei Wang, Jiaya Jia},
  booktitle = {ECCV},
  year      = {2022}
}
```