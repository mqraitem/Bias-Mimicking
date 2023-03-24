import logging
import os
import pickle
from pathlib import Path

import PIL
import numpy as np
import torch
import torch.utils.data
from debias.datasets.utils import TwoCropTransform, get_confusion_matrix
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from debias.datasets.sampling_dataset import SamplingDataset

class UTKFace:
    def __init__(self, root, transform, **kwargs):
        self.root = root
        self.filenames = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        X = PIL.Image.open(os.path.join(self.root, filename))
        age = int(filename.split('_')[0])

        if self.transform is not None:
            X = self.transform(X)

        return X, age


    def __len__(self):
        return len(self.filenames)


class BiasedUTKFace(SamplingDataset):
    def __init__(self, root, transform, split,
                 bias_attr='race', bias_rate=0.9,
                 sampling=False, 
                 diff_analysis=0.0, 
                 **kwargs):
        self.root = Path(root) / 'images'
        filenames = np.array(os.listdir(self.root))
        np.random.shuffle(filenames)
        num_files = len(filenames)
        num_train = int(num_files * 0.8)
        target_attr = 'gender'

        self.transform = transform
        self.target_attr = target_attr
        self.bias_rate = bias_rate
        self.bias_attr = bias_attr
        self.train = split == 'train'

        save_path = Path(root) / 'pickles' / f'biased_utk_face-target_{target_attr}-bias_{bias_attr}-{bias_rate}'
        if save_path.is_dir():
            print(f'use existing biased_utk_face from {save_path}')
            data_split = 'train' if self.train else 'test'
            self.files, self.targets, self.bias_targets = pickle.load(open(save_path / f'{data_split}_dataset.pkl', 'rb'))
            if split in ['valid', 'test']:
                save_path = Path(f'clusters/utk_face_rand_indices_{bias_attr}.pkl')
                if not save_path.exists():
                    rand_indices = torch.randperm(len(self.targets))
                    pickle.dump(rand_indices, open(save_path, 'wb'))
                else:
                    rand_indices = pickle.load(open(save_path, 'rb'))
                num_total = len(rand_indices)
                num_valid = int(0.5 * num_total)
                
                if split == 'valid':
                    indices = rand_indices[:num_valid]
                elif split == 'test':
                    indices = rand_indices[num_valid:]
                
                indices = indices.numpy()
                
                self.files = self.files[indices]
                self.targets = self.targets[indices]
                self.bias_targets = self.bias_targets[indices]
        else:
            train_dataset = self.build(filenames[:num_train], train=True)
            test_dataset = self.build(filenames[num_train:], train=False)

            print(f'save biased_utk_face to {save_path}')
            save_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(train_dataset, open(save_path / f'train_dataset.pkl', 'wb'))
            pickle.dump(test_dataset, open(save_path / f'test_dataset.pkl', 'wb'))

            self.files, self.targets, self.bias_targets = train_dataset if self.train else test_dataset
        
        self.targets, self.bias_targets = torch.from_numpy(self.targets).long(), torch.from_numpy(
            self.bias_targets).long()

        self.set_dro_info()
        self.calculate_bias_weights()
        self.targets_bin = self.get_targets_bin()
        
        self.samples_check = torch.zeros((len(self.targets))) 
        self.set_main_data()
        self.eye_tsr = self.get_eye_tsr()

        print("Distribution Before Sampling: ")
        self.print_new_distro()

        if split == 'train' and sampling == 'bin': 
            self.bias_mimick()

        if split == 'train' and sampling == 'analysis':
            self.under_sample_bin(diff=diff_analysis) 
                            
        if split == 'train' and sampling == 'ce': 
            self.under_sample_ce() 
        
        if split == 'train' and sampling == 'os': 
            self.over_sample_ce() 

        if split == 'train':
            print("Distribution After Sampling: ")
            self.print_new_distro()

                
        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                                          targets=self.targets,
                                                                                                          biases=self.bias_targets)
        
        print(f'Use BiasedUTKFace - target_attr: {target_attr}')

        print(
            f'BiasedUTKFace -- total: {len(self.files)}, target_attr: {self.target_attr}, bias_attr: {self.bias_attr} ' \
            f'bias_rate: {self.bias_rate}')

        print(
            [f'[{split}] target_{i}-bias_{j}: {sum((self.targets == i) & (self.bias_targets == j))}' for i in (0, 1) for
             j in (0, 1)])


    def set_to_keep(self, to_keep_idx): 
        self.targets = self.targets[to_keep_idx] 
        self.bias_targets = self.bias_targets[to_keep_idx] 
        self.targets_bin = self.targets_bin[to_keep_idx]
        self.files = self.files[to_keep_idx] 
        self.group_weights = self.group_weights[to_keep_idx]
        self.groups_idx = self.groups_idx[to_keep_idx]

    def reset_data(self): 
        self.targets = torch.clone(self.targets_main)
        self.bias_targets = torch.clone(self.bias_targets_main)
        self.targets_bin = torch.clone(self.targets_bin_main)
        self.group_weights = torch.clone(self.group_weights_main)
        self.groups_idx = torch.clone(self.groups_idx_main)
        self.files = np.copy(self.files_main)

    def set_main_data(self): 
        self.targets_main = torch.clone(self.targets)
        self.bias_targets_main = torch.clone(self.bias_targets)
        self.targets_bin_main = torch.clone(self.targets_bin)
        self.group_weights_main = torch.clone(self.group_weights)
        self.groups_idx_main = torch.clone(self.groups_idx)
        self.files_main = np.copy(self.files)
        
    def build(self, filenames, train=False):
        
        attr_dict = {
            'age': (0, lambda x: x >= 20, lambda x: x <= 10,),
            'gender': (1, lambda x: x == 0, lambda x: x == 1),
            'race': (2, lambda x: x == 0, lambda x: x != 0),
        }
        assert self.target_attr in attr_dict.keys()
        target_cls_idx, *target_filters = attr_dict[self.target_attr]
        bias_cls_idx, *bias_filters = attr_dict[self.bias_attr]

        target_classes = self.get_class_from_filename(filenames, target_cls_idx)
        bias_classes = self.get_class_from_filename(filenames, bias_cls_idx)

        total_files = []
        total_targets = []
        total_bias_targets = []

        for i in (0, 1):
            major_idx = np.where(target_filters[i](target_classes) & bias_filters[i](bias_classes))[0]
            minor_idx = np.where(target_filters[1 - i](target_classes) & bias_filters[i](bias_classes))[0]
            np.random.shuffle(minor_idx)

            num_major = major_idx.shape[0]
            num_minor_org = minor_idx.shape[0]
            if train:
                num_minor = int(num_major * (1 - self.bias_rate))
            else:
                num_minor = minor_idx.shape[0]
            num_minor = min(num_minor, num_minor_org)
            num_total = num_major + num_minor

            majors = filenames[major_idx]
            minors = filenames[minor_idx][:num_minor]

            total_files.append(np.concatenate((majors, minors)))
            total_bias_targets.append(np.ones(num_total) * i)
            total_targets.append(np.concatenate((np.ones(num_major) * i, np.ones(num_minor) * (1 - i))))

        files = np.concatenate(total_files)
        targets = np.concatenate(total_targets)
        bias_targets = np.concatenate(total_bias_targets)
        return files, targets, bias_targets


    def get_class_from_filename(self, filenames, cls_idx):
        return np.array([int(fname.split('_')[cls_idx]) if len(fname.split('_')) == 4 else 10 for fname in filenames])

    def __getitem__(self, index):
        filename, target, bias, target_bin = self.files[index], int(self.targets[index]), int(self.bias_targets[index]), self.targets_bin[index]
        
        gc = self.group_weights[index]
        X = PIL.Image.open(os.path.join(self.root, filename))

        if self.transform is not None:
            X = self.transform(X)

        group_idx = self.groups_idx[index]
        return X, target, target_bin, bias, gc, group_idx

    def __len__(self):
        return len(self.files)


def get_utk_face(root, batch_size, split, bias_attr='race', bias_rate=0.9, num_workers=4,
                  aug=False, image_size=64, 
                  two_crop=False, 
                  ratio=0, 
                  given_y=True,
                  sampling=False, 
                  repair=False, 
                  diff_analysis=0.0, 
                  reweight_sampler=False):

    logging.info(f'get_utk_face - split: {split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}')
    size_dict = {64: 72, 128: 144, 224: 256}
    load_size = size_dict[image_size]
    train = split == 'train'

    if train:
        if aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    if two_crop:
        transform = TwoCropTransform(transform)

    dataset = BiasedUTKFace(root, transform=transform, split=split, bias_rate=bias_rate, bias_attr=bias_attr, sampling=sampling, diff_analysis=diff_analysis)

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    #Determine whether samples has to be used. 
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

    #Determine shuffler value
    if repair:
        shuffle=False 

    elif sampler is None:
        shuffle = True 
    else: 
        shuffle = False


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=two_crop
    )

    return dataloader
