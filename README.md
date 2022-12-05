# Bias Mimicking: A simple sampling approach for Bias Mitigation 

Official Pytorch implementation of [Bias Mimicking: A simple sampling approach for Bias Mitigation](https://arxiv.org/pdf/2209.15605.pdf). 

## Setup

### Set up conda environment  
```
conda create -n bias_mimicking python=3.8
conda activate bias_mimicking
```

### Install packages

* pytorch=1.10.1 
* scipy
* tqdm 
* scikit-learn

### Prepare dataset.

- CelebA  
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset under `data/celeba`

- UTKFace  
Download [UTKFace](https://susanqq.github.io/UTKFace/) dataset under `data/utk_face`

- CIFAR10  
Download [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset under `data/cifar10`


As discussed in the paper, we train on subsampled versions of CelebA and UTKFace. The information required to reproduce the the splits are in data/[DATASET]/pickles. The code will automatically load the right splits. 

## Train.

From the main directory, run: 

```
python train_[DATASET]/train_[DATASET]_[METHOD].py --seed [SEED]
```

To train our method on celeba, run: 

```
python train_celeba/train_celeba_bm.py --mode [none/us/uw/os] --seed [SEED]
```

where mode refers to whether the distribution is left as is/undersampled/upweighted/oversampled when training the predictive linear layer. 

## Acknowledgements

The code for non sampling methods builds on [this work](https://github.com/grayhong/bias-contrastive-learning). Furthermore, the code for GroupDRO is obtained from [this work](https://github.com/kohpangwei/group_DRO)