import argparse
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(1, './')

from debias.datasets.celeba import get_celeba
from debias.networks.resnet import FCResNet18_Base
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                pretty_dict, save_model, set_seed)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='blonde')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--training_ratio', type=float, default=2)
    parser.add_argument('--alpha', type=float, default=1)

    opt = parser.parse_args()

    return opt


def set_model():
    model = FCResNet18_Base().cuda()
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    class_network = nn.Linear(512, 2).cuda()
    domain_network = nn.Linear(512, 2).cuda()

    return [model, class_network, domain_network], [class_criterion, domain_criterion]


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model[0].train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    for images, labels, _, biases, _, _ in tqdm(train_iter, ascii=True):
        bsz = labels.shape[0]
        labels, biases = labels.cuda(), biases.cuda()
        
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        
        images = images.cuda()
        features = model[0](images)
        class_out = model[1](features)
        domain_out = model[2](features)

        class_loss = criterion[0](class_out, labels)
        domain_loss = criterion[1](domain_out, biases) 

        if epoch % opt.training_ratio == 0:
            log_softmax = F.log_softmax(domain_out, dim=1)
            confusion_loss = -log_softmax.mean(dim=1).mean()
            loss = class_loss + opt.alpha*confusion_loss
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()
        
        else:
            # Update the domain classifier
            domain_loss.backward()
            optimizer[2].step()

        avg_loss.update(class_loss.item(), bsz)

    return avg_loss.avg


def validate(val_loader, model):
    model[0].eval()

    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))

    with torch.no_grad():
        for images, labels, _, biases, _, _ in val_loader:
            images, labels, biases = images.cuda(), labels.cuda(), biases.cuda()
            bsz = labels.shape[0]

            feats = model[0](images)
            output = model[1](feats)

            preds = output.data.max(1, keepdim=True)[1].squeeze(1)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_mean(), attrwise_acc_meter.get_acc_diff() 


def main():
    opt = parse_option()

    exp_name = f'adv-celeba_{opt.task}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = './data/celeba'
    train_loader = get_celeba(
        root,
        batch_size=opt.bs,
        target_attr=opt.task,
        split='train',
        aug=False)

    val_loaders = {}
    val_loaders['valid'] = get_celeba(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='train_valid',
        aug=False)

    val_loaders['test'] = get_celeba(
        root,
        batch_size=256,
        target_attr=opt.task,
        split='valid',
        aug=False)

    model, criterion = set_model()

    decay_epochs = [opt.epochs // 3, opt.epochs * 2 // 3]

    optimizer_base = torch.optim.Adam(model[0].parameters(), lr=opt.lr, weight_decay=1e-4)
    optimizer_class = torch.optim.Adam(model[1].parameters(), lr=opt.lr, weight_decay=1e-4)
    optimizer_domain = torch.optim.Adam(model[2].parameters(), lr=opt.lr, weight_decay=1e-4)
    optimizers = [optimizer_base, optimizer_class, optimizer_domain]

    scheduler_base = torch.optim.lr_scheduler.MultiStepLR(optimizers[0], milestones=decay_epochs, gamma=0.1)
    scheduler_class = torch.optim.lr_scheduler.MultiStepLR(optimizers[1], milestones=decay_epochs, gamma=0.1)
    scheduler_domain = torch.optim.lr_scheduler.MultiStepLR(optimizers[2], milestones=decay_epochs, gamma=0.1)
    schedulers = [scheduler_base,scheduler_class, scheduler_domain]

    logging.info(f"decay_epochs: {decay_epochs}")

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {schedulers[0].get_last_lr()[0]}')
        loss = train(train_loader, model, criterion, optimizers, epoch, opt)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss:.4f}')

        schedulers[0].step()
        schedulers[1].step()
        schedulers[2].step()

        stats = pretty_dict(epoch=epoch)
        for key, val_loader in val_loaders.items():
            accs, valid_attrwise_accs, diff = validate(val_loader, model)

            stats[f'{key}/acc'] = accs.item()
            stats[f'{key}/acc_unbiased'] = torch.mean(valid_attrwise_accs).item() * 100
            stats[f'{key}/diff'] = diff.item() * 100
            
            eye_tsr = train_loader.dataset.eye_tsr 
            
            stats[f'{key}/acc_skew'] = valid_attrwise_accs[eye_tsr > 0.0].mean().item() * 100
            stats[f'{key}/acc_align'] = valid_attrwise_accs[eye_tsr == 0.0].mean().item() * 100

        logging.info(f'[{epoch} / {opt.epochs}] {valid_attrwise_accs} {stats}')
        for tag in val_loaders.keys():
            if stats[f'{tag}/acc_unbiased'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc_unbiased']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'best_{tag}_{k}': v for k, v in stats.items()})

            logging.info(
                f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

if __name__ == '__main__':
    main()
