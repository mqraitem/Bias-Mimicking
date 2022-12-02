# Bias Mimicking: A simple sampling approach for Bias Mitigation 

Official Pytorch implementation of Bias Mimicking: A simple sampling approach for Bias Mitigation

## Setup

1. Set up conda environment  
```
conda create -n bias_mimicking python=3.8
conda activate biasm-bias_mimicking
```

2. Install packages

Install the following packages: 

* pytorch=1.9.0 torchvision==0.10.0 
* scipy
* tqdm 

3. Prepare dataset.

- CelebA  
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset under `data/celeba`

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) dataset under `data/utk_face`

- CIFAR10 
Download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

As discussed in the paper, we train on subsampled versions of CelebA and UTKFace. The information required to reproduce the the splits are in data/[DATASET]/pickles. The code will automatically load the right splits. 

## Train.

From the main directory, run: 

```
python train_[DATASET]/train_[DATASET]_[METHOD].py --seed [SEED]
```

