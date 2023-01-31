import torch
import numpy as np 

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(len(np.unique(biases)), num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, len(np.unique(biases)))
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1
    
    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by


def get_unsup_confusion_matrix(num_classes, targets, biases, marginals):
    confusion_matrix_org = torch.zeros(num_classes, num_classes).float()
    confusion_matrix_cnt = torch.zeros(num_classes, num_classes).float()
    for t, p, m in zip(targets, biases, marginals):
        confusion_matrix_org[p.long(), t.long()] += m
        confusion_matrix_cnt[p.long(), t.long()] += 1

    zero_idx = confusion_matrix_org == 0
    confusion_matrix_cnt[confusion_matrix_cnt == 0] = 1
    confusion_matrix_org = confusion_matrix_org / confusion_matrix_cnt
    confusion_matrix_org[zero_idx] = 1
    confusion_matrix_org = 1 / confusion_matrix_org
    confusion_matrix_org[zero_idx] = 0

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix


import numpy as np 

def get_samples_counts(all_labels_nb, all_bias):
    g_idxs = [] 
    g_counts = [] 
    full_idx = np.arange(len(all_bias))

    num_targets = len(np.unique(all_labels_nb)) 
    num_biases = len(np.unique(all_bias))
     
    for i in range(num_biases): 
        for j in range(num_targets): 
            g_idxs.append(full_idx[np.logical_and(all_bias == i, all_labels_nb == j)])
            g_counts.append(len(g_idxs[-1])) 
    return g_idxs, g_counts

def under_sample_features(all_bias, all_feats, all_labels_nb):

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    min_group = min(g_counts) 

    to_keep_idx_all = [] 
    for _, group_idx in enumerate(g_idxs): 
        to_keep_idx = np.random.choice(group_idx, min_group)
        to_keep_idx_all.extend(to_keep_idx)

    all_feats = all_feats[to_keep_idx_all]
    all_labels_nb = all_labels_nb[to_keep_idx_all]
    all_bias = all_bias[to_keep_idx_all]

    full_idx = np.arange(len(all_feats))
    np.random.shuffle(full_idx)

    all_feats = all_feats[full_idx]
    all_labels_nb = all_labels_nb[full_idx]
    all_bias = all_bias[full_idx]

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    return all_feats, all_labels_nb


def over_sample_features(all_bias, all_feats, all_labels_nb):

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)
    max_group = max(g_counts) 

    for idx, group_idx in enumerate(g_idxs): 
        to_add = max_group - len(group_idx)
        to_add_idx = np.random.choice(group_idx, to_add)

        if to_add == 0: 
            continue

        all_feats = np.concatenate((all_feats, all_feats[to_add_idx]), axis=0) 
        all_labels_nb = np.concatenate((all_labels_nb, all_labels_nb[to_add_idx]), axis=0) 
        all_bias = np.concatenate((all_bias, all_bias[to_add_idx]), axis=0) 

    full_idx = np.arange(len(all_feats))
    np.random.shuffle(full_idx)

    all_feats = all_feats[full_idx]
    all_labels_nb = all_labels_nb[full_idx]
    all_bias = all_bias[full_idx]

    g_idxs, g_counts = get_samples_counts(all_labels_nb, all_bias)

    return all_feats, all_labels_nb