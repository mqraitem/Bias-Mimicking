from __future__ import print_function

import logging
import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from sklearn.metrics import average_precision_score


class MultiDimAverageMeter(object):
    def __init__(self, dims=(2, 2)):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.bias = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):

        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )

        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )
    
    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def get_acc_diff(self): 
        unbiased_acc = self.cum/self.cnt
        diff = unbiased_acc[self.idx_helper[:, 0]] - unbiased_acc[self.idx_helper[:,1]]
        diff = torch.abs(diff)
        diff = diff.mean()
        return diff

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_weighted_AP(target, predict_prob, class_weight_list):
    per_class_AP = []
    for i in range(target.shape[1] - 1):
        class_weight = target[:, i]*class_weight_list[i] \
                       + (1-target[:, i])*np.ones(class_weight_list[i].shape)
        per_class_AP.append(average_precision_score(target[:, i], predict_prob[:, i], 
                                sample_weight=class_weight))
        
    return per_class_AP

def compute_class_weight(target):
    domain_label = target[:, -1]
    per_class_weight = []

    for i in range(target.shape[1]):
        class_label = target[:, i]
        cp = class_label.sum() # class is positive
        cn = target.shape[0] - cp # class is negative
        cn_dn = ((class_label + domain_label)==0).sum() # class is negative, domain is negative
        cn_dp = ((class_label - domain_label)==-1).sum()
        cp_dn = ((class_label - domain_label)==1).sum()
        cp_dp = ((class_label + domain_label)==2).sum()

        per_class_weight.append(
            (class_label*cp + (1-class_label)*cn) / 
                (2*(
                    (1-class_label)*(1-domain_label)*cn_dn
                    + (1-class_label)*domain_label*cn_dp
                    + class_label*(1-domain_label)*cp_dn
                    + class_label*domain_label*cp_dp
                   )
                )
        )
    return per_class_weight


def set_seed(seed):
    logging.info(f'=======> Using Fixed Random Seed: {seed} <========')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    
    #torch.use_deterministic_algorithms(True) 
    cudnn.benchmark = False  # set to False for final report


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class pretty_dict(dict):                                              
    def __str__(self):
        return str({k: round(v, 3) if isinstance(v,float) else v for k, v in self.items()})



