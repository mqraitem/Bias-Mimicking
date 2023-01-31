import logging
import pickle
from pathlib import Path

import torch
import numpy as np
from debias.datasets.utils import TwoCropTransform, get_confusion_matrix
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA
from debias.datasets.sampling_dataset import SamplingDataset

class BiasedCelebASplit(SamplingDataset):
    def __init__(self, root, split, transform, target_attr, under_sample, diff_analysis, **kwargs):
        self.transform = transform
        self.target_attr = target_attr
        
        self.celeba = CelebA(
            root=root,
            split="train" if split == "train_valid" else split,
            target_type="attr",
            transform=transform,
        )

        self.bias_idx = 20 #Gender Attribute. 
        
        if target_attr == 'blonde':
            self.target_idx = 9
            if split in ['train', 'train_valid'] :
                save_path = Path(root) / 'pickles' / 'blonde'
                if save_path.is_dir():
                    print(f'use existing blonde indices from {save_path}')
                    self.indices = pickle.load(open(save_path / 'indices.pkl', 'rb'))
                else:
                    self.indices = self.build_blonde()
                    print(f'save blonde indices to {save_path}')
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f'indices.pkl', 'wb'))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))

        
        elif target_attr == 'makeup':
            self.target_idx = 18
            self.attr = self.celeba.attr
            self.indices = torch.arange(len(self.celeba))
        
        elif target_attr == 'black':
            self.target_idx = 8
            if split in ['train', 'train_valid'] :
                save_path = Path(root) / 'pickles' / 'black'
                if save_path.is_dir():
                    print(f'use existing black indices from {save_path}')
                    self.indices = pickle.load(open(save_path / 'indices.pkl', 'rb'))
                else:
                    self.indices = self.build_black()
                    print(f'save black indices to {save_path}')
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f'indices.pkl', 'wb'))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))

        elif target_attr == 'smiling':
            self.target_idx = 31
            if split in ['train', 'train_valid'] :
                save_path = Path(root) / 'pickles' / 'smiling'
                if save_path.is_dir():
                    print(f'use existing smiling indices from {save_path}')
                    self.indices = pickle.load(open(save_path / 'indices.pkl', 'rb'))
                else:
                    self.indices = self.build_smiling()
                    print(f'save smiling indices to {save_path}')
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f'indices.pkl', 'wb'))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))
        
        else:
            raise AttributeError
            
        if split in ['train', 'train_valid']:
            
            rand_indices = torch.randperm(len(self.indices))
            
            num_total = len(rand_indices)
            num_train = int(0.8 * num_total)
            
            if split == 'train':
                indices = rand_indices[:num_train]
            elif split == 'train_valid':
                indices = rand_indices[num_train:]
            
            self.indices = self.indices[indices]
            self.attr = self.attr[indices]
        
        
        self.targets = self.attr[:, self.target_idx]
        self.bias_targets = self.attr[:, self.bias_idx]

        self.set_dro_info()
        self.calculate_bias_weights()
        self.targets_bin = self.get_targets_bin()
        
        self.samples_check = torch.zeros((len(self.targets))) 
        self.set_main_data()
        self.eye_tsr = self.get_eye_tsr()

        if split == 'train' and under_sample == 'bin': 
            self.bias_mimick()
        
        if split == 'train' and under_sample == 'ce': 
            self.under_sample_ce() 
        
        if split == 'train' and under_sample == 'analysis':
            self.under_sample_bin(diff=diff_analysis) 
                    
        if split == 'train' and under_sample == 'os': 
            self.over_sample_ce() 

        if split == 'train':
            print("Distribution After Sampling: ")
            self.print_new_distro()

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                                          targets=self.targets,
                                                                                                          biases=self.bias_targets)
                                                                                                          
        print(f'Use BiasedCelebASplit \n target_attr: {target_attr} split: {split} \n {self.confusion_matrix_org}')

    def set_to_keep(self, to_keep_idx): 
        self.targets = self.targets[to_keep_idx] 
        self.targets_bin = self.targets_bin[to_keep_idx]
        self.bias_targets = self.bias_targets[to_keep_idx] 
        self.group_weights = self.group_weights[to_keep_idx]
        self.groups_idx = self.groups_idx[to_keep_idx]
        self.indices = self.indices[to_keep_idx]

    def build_black(self):
        bias_targets = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(bias_targets == 0) & (targets == 1)]
        non_selects = torch.arange(len(self.celeba))[~((bias_targets == 0) & (targets == 1))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:10000], non_selects])
        return indices

    def build_blonde(self):
        bias_targets = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(bias_targets == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((bias_targets == 0) & (targets == 0))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def build_smiling(self):
        bias_targets = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]

        selects_1 = torch.arange(len(self.celeba))[(bias_targets == 0) & (targets == 0)]
        selects_2 = torch.arange(len(self.celeba))[(bias_targets == 1) & (targets == 1)]
        non_selects = torch.arange(len(self.celeba))[~(torch.logical_or((bias_targets == 0) & (targets == 0), 
                                                                        (bias_targets == 1) & (targets == 1)))]

        indices = torch.cat([selects_1[:10000], selects_2[:10000], non_selects])
        return indices

    def set_main_data(self): 
        self.targets_main = torch.clone(self.targets)
        self.targets_bin_main = torch.clone(self.targets_bin)
        self.bias_targets_main = torch.clone(self.bias_targets)
        self.group_weights_main = torch.clone(self.group_weights)
        self.groups_idx_main = torch.clone(self.groups_idx)
        self.indices_main = torch.clone(self.indices)

    def reset_data(self): 
        self.targets = torch.clone(self.targets_main)
        self.bias_targets = torch.clone(self.bias_targets_main)
        self.targets_bin = torch.clone(self.targets_bin_main)
        self.group_weights = torch.clone(self.group_weights_main)
        self.groups_idx = torch.clone(self.groups_idx_main)
        self.indices = torch.clone(self.indices_main)

    def __getitem__(self, index):
        img, _ = self.celeba.__getitem__(self.indices[index])
        target, bias = self.targets[index], self.bias_targets[index]
        target_bin = self.targets_bin[index]
        gc = self.group_weights[index] 
        groups_idx = self.groups_idx[index]
        
        return img, target, target_bin, bias, gc, groups_idx

    def __len__(self):
        return len(self.targets)


def get_celeba(root, batch_size, target_attr='blonde', split='train', num_workers=2, aug=True, two_crop=False, ratio=0,
               img_size=224, 
               given_y=True, 
               under_sample=None, 
               diff_analysis = 0.0,
               reweight_sampler=False):

    logging.info(f'get_celeba - split:{split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}')
    if split == 'eval':
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        if aug:
            transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        else:
            transform = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    if two_crop:
        transform = TwoCropTransform(transform)

    dataset = BiasedCelebASplit(
        root=root,
        split=split,
        transform=transform,
        target_attr=target_attr,
        under_sample=under_sample,
        diff_analysis = diff_analysis,
    )
    

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)
    
    if ratio != 0:
        if given_y:
            weights = [1 / dataset.confusion_matrix_by[c, b] for c, b in zip(dataset.targets, dataset.bias_targets)]
        else:
            weights = [1 / dataset.confusion_matrix[b, c] for c, b in zip(dataset.targets, dataset.bias_targets)]
        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    elif reweight_sampler:
        weights = len(dataset)/dataset.group_counts()
        weights = weights[dataset.groups_idx.long()]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=two_crop
    )
    return dataloader
