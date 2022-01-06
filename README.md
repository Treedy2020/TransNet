## Overview

This is the PyTorch implementation of paper "TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System".

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading

We appologize for that due to the oversight of our earlier experiments, we didn't save the complete Checkpoints results, we will upload our checkpoints under different scenarios in [Google Drive](https://drive.google.com/drive/folders/10AxRFCE1Nbiqc0JgcFdQZ8mxQV8YbR8F?usp=sharing) as soon as possible.  You can still check the authenticity of our results by training a new TransNet yourself and see its performance, the test NMSE and training MSE loss will be printed during your training. A 400 epochs training dosen't take very long (about 3 and half hours on a single RTX 2060), and you are able to reproduce any results in  Table 1 of our paper.

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
│   │     ├── in04.pth
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

The main results reported in our paper are presented as follows. All the listed results can be found in Table1 of our paper. They are achieved from training TransNet with our  2 kind of training scheme ( constant learning rate at 1e-4 for 400/2500 epochs).

Results of 400 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--: 
indoor | 1/4 | -29.22 | 35.72M 
indoor | 1/8 | -21.62 | 34.70M 
indoor | 1/16 | -14.98 | 34.14M 
indoor | 1/32 | -9.83 | 33.88M 
indoor | 1/64 | -6.05 | 33.75M 
outdoor | 1/4 | -13.99 | 35.72M 
outdoor | 1/8 | -9.57 | 34.70M 
outdoor | 1/16 | -6.90 | 34.14M 
outdoor | 1/32 | -3.30 | 33.88M 
outdoor | 1/64 | -2.20 | 33.75M 

Results of 2500 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--: 
indoor | 1/4 | -33.12 | 35.72M 
indoor | 1/8 | -22.91 | 34.70M 
indoor | 1/16 | -15.00 | 34.14M 
indoor | 1/32 | -10.49 | 33.88M 
indoor | 1/64 | -6.66 | 33.75M 
outdoor | 1/4 | -14.86 | 35.72M 
outdoor | 1/8 | -9.99 | 34.70M 
outdoor | 1/16 | -7.82 | 34.14M 
outdoor | 1/32 | -4.42 | 33.88M 
outdoor | 1/64 | -2.62 | 33.75M 



As aforementioned, we can not provide model checkpoints for the results temporarily. We will improve this in the next version of our codes, sorry for the time being you need to train your TransNet to test its performance.



## Acknowledgment

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 


Thanks two open source works, CRNet and CLNet, that build on work above and advance the CSI feedback problem in DL, you can find their related work in [Github-Python-PyTorch CRNet](https://github.com/Kylin9511/CRNet) and [Github-Python-PyTorch CLNet](https://github.com/SIJIEJI/CLNet)

