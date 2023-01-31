import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import sys
sys.path.insert(1, './')

from debias.datasets.celeba import get_celeba
from debias.datasets.utils import over_sample_features, under_sample_features
from debias.networks.resnet import FCResNet18
from debias.utils.logging import set_logging
from debias.utils.utils import (AverageMeter, MultiDimAverageMeter, accuracy,
                                pretty_dict, save_model, set_seed)
from sklearn.linear_model import LogisticRegression


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task', type=str, default='blonde')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--bs', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_layer', type=float, default=1e-3)

    parser.add_argument('--mode', type=str, default='none')

    opt = parser.parse_args()

    return opt

def set_model():
    model = FCResNet18(num_classes=2).cuda()
    pred = nn.Linear(512, 2).cuda()

    models = { 
                'model': model, 
                'pred': pred 
             }

    criterion = { 
                'bin': nn.BCEWithLogitsLoss(reduction='none'),
                'multi': nn.CrossEntropyLoss(reduction='none')
                } 

    return models, criterion


def train(train_loader, model, criterion, optimizer_model, opt, optimizer_layer):
    model['model'].train()
    avg_loss = AverageMeter()

    train_iter = iter(train_loader)
    
    all_outputs = [] 
    all_labels_nb = [] 
    all_gc = [] 
    all_feats = [] 
    all_bias = [] 

    sig = nn.Sigmoid()

    for images, labels, labels_bin, biases, gc, _ in  tqdm(train_iter, ascii=True):
         
        bsz = labels.shape[0]
        labels, biases, labels_bin = labels.cuda(), biases.cuda(), labels_bin.cuda()
    
        images = images.cuda()
        logits, feat = model['model'](images)

        multi = torch.ones_like(labels_bin) 

        multi[labels_bin == -1] = 0 
        labels_bin[labels_bin == -1] = 0 

        loss = criterion['bin'](logits, labels_bin)
        loss = loss*multi 

        div = torch.sum(multi) 
        loss = torch.sum(loss/div)

        avg_loss.update(loss.item(), bsz)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        all_outputs.append(sig(logits).cpu().detach().numpy()) 
        all_labels_nb.append(labels.cpu().detach().numpy())
        all_gc.append(gc.numpy())
        all_bias.append(biases.cpu().detach().numpy())
        all_feats.append(feat.cpu().detach().numpy())
    
    all_labels_nb = np.concatenate(all_labels_nb, axis=0) 
    all_gc = np.concatenate(all_gc, axis=0) 
    all_bias = np.concatenate(all_bias, axis=0) 
    all_feats = np.concatenate(all_feats, axis=0) 

    if opt.mode == 'os':
        all_feats, all_labels_nb = over_sample_features(all_bias, all_feats, all_labels_nb)

    elif opt.mode == 'us':
        all_feats, all_labels_nb = under_sample_features(all_bias, all_feats, all_labels_nb)

    batch_size = opt.bs 
    total_samples = len(all_labels_nb)
    num_batches = total_samples//batch_size 

    all_idx = np.arange(total_samples)
    np.random.shuffle(all_idx)

    all_feats[all_idx] = all_feats 
    all_labels_nb[all_idx] = all_labels_nb
    if opt.mode == 'uw':
        all_gc[all_idx] = all_gc

    for batch_idx in range(num_batches): 
        start = batch_idx * batch_size 
        end = min(total_samples, start + batch_size)

        feats = torch.from_numpy(all_feats[start:end]).cuda()
        labels = torch.from_numpy(all_labels_nb[start:end]).cuda()
        gc = torch.from_numpy(all_gc[start:end]).cuda()

        optimizer_layer.zero_grad() 
        out_lr = model['pred'](feats)

        if opt.mode == 'uw':
            loss = criterion['multi'](out_lr, labels)*gc
            loss = torch.mean(loss)
        else: 
            loss = criterion['multi'](out_lr, labels)
            loss = torch.mean(loss)
            
        loss.backward() 
        optimizer_layer.step()
        
    return avg_loss.avg


def validate(val_loader, model):
    model['model'].eval()
    
    top1 = AverageMeter()
    attrwise_acc_meter = MultiDimAverageMeter(dims=(2, 2))
    
    with torch.no_grad():
        for idx, (images, labels, _, biases, _, _) in enumerate(tqdm(val_loader, ascii=True)):
            images = images.cuda()
            bsz = labels.shape[0]
            
            output,feats = model['model'](images)
            output = model['pred'](feats).detach().cpu()
            preds = output.data.max(1, keepdim=True)[1].squeeze(1)
            
            acc1, = accuracy(output, labels.cpu(), topk=(1,))
            top1.update(acc1[0], bsz)

            corrects = (preds == labels).long()
            attrwise_acc_meter.add(corrects.cpu(), 
                                  torch.stack([labels.cpu(), biases.cpu()], dim=1))

    return top1.avg, attrwise_acc_meter.get_mean(), attrwise_acc_meter.get_acc_diff() 


def main():
    opt = parse_option()

    exp_name = f'bm-celeba_{opt.task}-{opt.exp_name}-lr{opt.lr}-bs{opt.bs}-seed{opt.seed}'
    opt.exp_name = exp_name
    
    if opt.task == "makeup":
        opt.epochs = 40
    elif opt.task == "blonde":
        opt.epochs = 10
    elif opt.task == "black":
        opt.epochs = 10
    elif opt.task == "smiling":
        opt.epochs = 10
    else:
        raise AttributeError()

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
        aug=False,
        under_sample = 'bin')

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

    optimizer = torch.optim.Adam(model['model'].parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    optimizer_layer =  torch.optim.Adam(model['pred'].parameters(), lr=opt.lr_layer, weight_decay=1e-4)
    scheduler_layer = torch.optim.lr_scheduler.MultiStepLR(optimizer_layer, milestones=decay_epochs, gamma=0.1)


    logging.info(f"decay_epochs: {decay_epochs}")
    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train(train_loader, model, criterion, optimizer, opt, optimizer_layer)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss:.4f}')

        scheduler.step()
        scheduler_layer.step()

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

                save_file = save_path / 'checkpoints' / f'best_{tag}.pth'
                # save_model(model, optimizer, opt, epoch, save_file)
            logging.info(
                f'[{epoch} / {opt.epochs}] best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \n best_stats: {best_stats[tag]}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    # save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
