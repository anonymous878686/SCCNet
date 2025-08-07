import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import datetime
from pathlib import Path


def write_log(log_file, tags, epoch, train_loss, train_acc, val_loss, top1_acc, top5_acc, learning_rate):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (f"{timestamp} | Epoch {epoch:03d} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} |"
                   f" val_loss: {val_loss:.4f} | top1_acc: {top1_acc:.4f} | top5_acc: {top5_acc:.4f} | learning_rate: {learning_rate:.4f}\n")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(log_message)


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    acc_loss=torch.zeros(1).to(device)
    acc_num=torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num=0
    data_loader=tqdm(train_loader,file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num+=images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        acc_num+=torch.eq(pred_classes, labels.to(device)).sum()
        loss=loss_func(pred, labels.to(device))
        loss.backward()
        acc_loss=acc_loss+loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               acc_loss.item() / (step + 1),
                                                                               acc_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return acc_loss.item() / (step + 1), acc_num.item() / sample_num


@torch.no_grad()
def evaluate(model, device, test_loader, epoch):
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    top1_correct = 0
    top5_correct = 0
    acc_loss = 0.0
    sample_num = 0

    test_loader = tqdm(test_loader, desc=f"[Test Epoch {epoch}]")

    for step, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).long()
        batch_size = images.size(0)
        sample_num += batch_size

        outputs = model(images)
        loss = loss_func(outputs, labels)
        acc_loss += loss.item()

        # Top-1
        _, pred_top1 = outputs.topk(1, dim=1, largest=True, sorted=True)
        top1_correct += (pred_top1.squeeze(1) == labels).sum().item()

        # Top-5 (with check for class count)
        num_classes = outputs.size(1)
        topk = min(5, num_classes)
        _, pred_top5 = outputs.topk(topk, dim=1, largest=True, sorted=True)
        top5_correct += sum([labels[i] in pred_top5[i] for i in range(batch_size)])

        test_loader.set_description(
            "[Test Epoch {}] Loss: {:.3f}, Top1: {:.2f}%, Top5: {:.2f}%".format(
                epoch, acc_loss / (step + 1), 100 * top1_correct / sample_num,
                100 * top5_correct / sample_num)
        )

    return (acc_loss / (step + 1), 
            top1_correct / sample_num, 
            top5_correct / sample_num)
