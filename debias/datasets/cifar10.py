import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np 
from PIL import Image
from tqdm import tqdm 
import random 

from debias.datasets.utils import TwoCropTransform, get_confusion_matrix, get_unsup_confusion_matrix
from torch.utils import data
from debias.datasets.sampling_dataset import SamplingDataset


class BiasedCifar10(SamplingDataset): 
    def __init__(self, root, transform, split, under_sample, corr, color_count, color_max): 
        self.transform = transform
        train_valid = (split == 'train')
        self.cifar10 = CIFAR10(root, train=train_valid, download=True)
        self.images = self.cifar10.data 
        self.targets = np.array(self.cifar10.targets) 
        self.bias_targets = np.zeros_like(self.targets) 
        self.split = split

        if not train_valid: 
            self.build_split() 
        
        if split == 'train':
            print('********') 
            print('Corr used %.2f'%corr)
            print('********') 

            self.corrupt_dataset(corr)
        
        else: 
            self.corrupt_test_dataset()
        
        self.targets, self.bias_targets = torch.from_numpy(self.targets).long(), torch.from_numpy(
            self.bias_targets).long()

        self.targets_bin = self.get_targets_bin()
        self.set_dro_info()
        self.calculate_bias_weights() 

        self.samples_check = torch.zeros((len(self.images))) 
        self.set_main_data()
        self.eye_tsr = self.get_eye_tsr()


        if under_sample == 'bin' and split == 'train': 
            self.bias_mimick()
        
        if under_sample == 'os' and split == 'train': 
            self.over_sample_ce()

        if under_sample == 'ce' and split == 'train': 
            self.under_sample_ce() 
        
        if under_sample == 'analysis' and split == 'train': 
            self.under_sample_analysis(color_count=color_count, color_max=color_max) 

        if split == 'train':
            print("Distribution After Sampling: ")
            self.print_new_distro()

        if not under_sample: 
          self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(
                num_classes=10,
                targets=self.targets,
                biases=self.bias_targets)

    def set_main_data(self):
        self.images_main = np.copy(self.images)
        self.targets_main = torch.clone(self.targets)
        self.bias_targets_main = torch.clone(self.bias_targets)
        self.targets_bin_main = torch.clone(self.targets_bin)
        self.group_weights_main = torch.clone(self.group_weights)
        self.groups_idx_main = torch.clone(self.groups_idx)

    def reset_data(self): 
        self.images = np.copy(self.images_main)
        self.targets = torch.clone(self.targets_main)
        self.bias_targets = torch.clone(self.bias_targets_main)
        self.targets_bin = torch.clone(self.targets_bin_main)
        self.group_weights = torch.clone(self.group_weights_main)
        self.groups_idx = torch.clone(self.groups_idx_main)

    def build_split(self):
        
        
        indices = {i:[] for i in range(10)} 
        size_per_class = 1000
        for idx, tar in enumerate(self.targets): 
            indices[tar].append(idx)

        if self.split == 'test': 
            start = 0 
            end = int(size_per_class * 0.9) 

        if self.split == 'valid':
            start = int(size_per_class * 0.9) 
            end = size_per_class

        final_indices = [] 
        for ind_group in indices.values(): 
            final_indices.extend(ind_group[start:end]) 
        
        random.shuffle(final_indices) 
        
        self.images = self.images[final_indices]
        self.bias_targets = self.bias_targets[final_indices]
        self.targets = self.targets[final_indices] 
        
    def rgb_to_grayscale(self, img):
        """Convert image to gray scale"""
        
        pil_img = Image.fromarray(img)
        pil_gray_img = pil_img.convert('L')
        np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
        np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])
    
        return np_gray_img

    def corrupt_test_dataset(self): 

        self.images_gray = np.copy(self.images)
        self.bias_targets_gray = np.copy(self.bias_targets)
        self.targets_gray = np.copy(self.targets)

        for idx, img in enumerate(self.images_gray): 
            self.images_gray[idx] = self.rgb_to_grayscale(img)
            self.bias_targets_gray[idx] = 1 

        self.images = np.concatenate((self.images, self.images_gray), axis=0)
        self.bias_targets = np.concatenate((self.bias_targets, self.bias_targets_gray), axis=0)
        self.targets = np.concatenate((self.targets, self.targets_gray), axis=0)


    def corrupt_dataset(self, skew_level):
        gray_classes = [0, 2, 4, 6, 8] 
        
        samples_by_class = {i:[] for i in range(10)} 
        for idx, target in enumerate(self.targets): 
            samples_by_class[target].append(idx) 

        for class_idx in tqdm(range(10), ascii=True): 
            class_samples = samples_by_class[class_idx]
            if class_idx in gray_classes: 
                samples_skew_num = int(len(class_samples) * skew_level) 
            else: 
                samples_skew_num = int(len(class_samples) * (1 - skew_level)) 
                
            samples_skew = random.sample(class_samples, samples_skew_num)
            for sample_idx in samples_skew: 
                self.images[sample_idx] = self.rgb_to_grayscale(self.images[sample_idx]) 
                self.bias_targets[sample_idx] = 1 

    def set_to_keep(self, to_keep_idx): 
        self.images = self.images_main[to_keep_idx]
        self.targets = self.targets_main[to_keep_idx] 
        self.bias_targets = self.bias_targets_main[to_keep_idx] 
        self.targets_bin = self.targets_bin_main[to_keep_idx]
        self.group_weights = self.group_weights_main[to_keep_idx]
        self.groups_idx = self.groups_idx_main[to_keep_idx]
    
    def __len__(self): 
        return len(self.images) 

    def __getitem__(self, index): 
        img = self.images[index]
        target = self.targets[index]
        bias = self.bias_targets[index]
        target_bin = self.targets_bin[index]
        gc = self.group_weights[index]
    
        if self.transform: 
            img = self.transform(img)
        
        group_idx = self.groups_idx[index]
        return img, target, target_bin, bias, gc, group_idx

def get_cifar10(root, split, num_workers=2, batch_size=128, 
                             aug=False, 
                             under_sample=False, 
                             shuffle=True,
                             two_crop=False, 
                             ratio=0,
                             corr=0.95, 
                             color_count = 1000, 
                             color_max=1, 
                             reweight_sampler=False): 

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    if split == 'test' or split=='valid': 
        train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    else:  
        if aug: 
            
            train_transform = transforms.Compose([
                                            transforms.ToPILImage(), 
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.RandomApply([
                                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                            ], p=0.8),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        else: 
            train_transform = transforms.Compose([
                                            transforms.ToPILImage(), 
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

    if two_crop:
        train_transform = TwoCropTransform(train_transform)
    
    dataset = BiasedCifar10(root, train_transform, split, under_sample, corr=corr, color_count = color_count, color_max=color_max) 
   
    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    if ratio != 0:
        
        weights = [1 / dataset.confusion_matrix_by[c, b] for c, b in zip(dataset.targets, dataset.bias_targets)]

        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))
        
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
    
    elif reweight_sampler:
        weights = len(dataset)/dataset.group_counts()
        weights = weights[dataset.groups_idx.long()]
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
   
    else:
        sampler = None
    
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None and split == 'train' else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False, 
        drop_last=two_crop)

    return dataloader 