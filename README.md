# Bias Mimicking: A simple sampling approach for Bias Mitigation 

Official Pytorch implementation of Bias Mimicking: A simple sampling approach for Bias Mitigation

## Setup

1. Install conda environment and activate it  
```
conda env create -f environment.yml
conda activate biasm-mimick
```

2. Prepare dataset.

- CelebA  
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset under `data/celeba`

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) dataset under `data/utk_face`

- CIFAR10 
Download [CIFAR10]()

As discussed in the paper, we train on subsampled versions of CelebA and UTKFace. The information required to reproduce the the splits are in data/[DATASET]/pickles. The code will automatically load the right splits. 

## Train.

From the main directory, run: 

```
python train_[DATASET]/train_[DATASET]_[METHOD].py --seed [SEED]
```

