## Overview

This is the PyTorch implementation of paper "TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System"(https://ieeexplore.ieee.org/document/9705497/keywords#keywords). You can cite our  paper by:

```
@ARTICLE{9705497,
  author={Cui, Yaodong and Guo, Aihuang and Song, Chunlin},
  journal={IEEE Wireless Communications Letters}, 
  title={TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LWC.2022.3149416}
```
## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [1.2 =< PyTorch <= 1.6](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading

 You can check  the performance of indoor and outdoor scenarios by downloading checkpoints in [Google Drive](https://drive.google.com/drive/folders/1eoxryQfrMOPVtbiMRdxXtp5KsBt13-hI?usp=sharing). We support more detail checpoints in  [Google Drive](https://drive.google.com/drive/folders/10AxRFCE1Nbiqc0JgcFdQZ8mxQV8YbR8F?usp=sharing). You can also check the authenticity of our results by training a new TransNet yourself and see its performance, the test NMSE and training MSE loss will be printed during your training. A 400 epochs training dosen't take very long (about 3 and half hours on a single RTX 2060), and you are able to reproduce TransNet-400ep results in  Table 1 of our paper.



#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── TransNet  # The cloned TransNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── 4_in.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Train TransNet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start  TransNet training from scratch. Change scenario by using `--scenario` . Change training epochs with '--epochs' and compression ratio with `--cr`.

``` bash
python /home/TransNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 400 \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --scheduler const \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction

The main results reported in our paper are presented as follows. All the listed results can be found in Table1 of our paper. They are achieved from training TransNet with our  2 kind of training scheme (constant learning rate at 1e-4 for 400/1000 epochs).

Results of 400 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--: 
indoor | 1/4 | -29.22 | 35.72M 
indoor | 1/8 | -21.62 | 34.70M 
indoor | 1/16 | -14.98 | 34.14M 
indoor | 1/32 | -9.83 | 33.88M 
indoor | 1/64 | -5.77 | 33.75M 
outdoor | 1/4 | -13.99 | 35.72M 
outdoor | 1/8 | -9.57 | 34.70M 
outdoor | 1/16 | -6.90 | 34.14M 
outdoor | 1/32 | -3.77 | 33.88M 
outdoor | 1/64 | -2.20 | 33.75M 

Results of 1000 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--: 
indoor | 1/4 | -32.38 | 35.72M 
indoor | 1/8 | -22.91 | 34.70M 
indoor | 1/16 | -15.00 | 34.14M 
indoor | 1/32 | -10.49 | 33.88M 
indoor | 1/64 | -6.08 | 33.75M 
outdoor | 1/4 | -14.86 | 35.72M 
outdoor | 1/8 | -9.99 | 34.70M 
outdoor | 1/16 | -7.82 | 34.14M 
outdoor | 1/32 | -4.13 | 33.88M 
outdoor | 1/64 | -2.62 | 33.75M 

**To reproduce all these results, simplely add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/TransNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/4_in.pth' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4\ # Note that cr should be same as  checkpoints
  --cpu \
  2>&1 | tee test_log.out

```




## Acknowledgment

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 


Thanks two open source works, CRNet and CLNet, that build on work above and advance the CSI feedback problem in DL, you can find their related work in [Github-Python-PyTorch CRNet](https://github.com/Kylin9511/CRNet) and [Github-Python-PyTorch CLNet](https://github.com/SIJIEJI/CLNet)

Thanks  the Github project members for the open source [Transformer tutorial](https://github.com/datawhalechina/Learn-NLP-with-Transformers), our base model for TransNet is based on their work.  

