#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    log_interval = CONFIG['log_interval']
    model.train()
    start = time()
    for i, data in enumerate(loader):
        users_b, bundles = data
        pred, bpr_loss, c_loss,graph_loss,ub_loss= model(users_b.to(device), bundles.to(device))
        # pred, bpr_loss, c_loss= model(users_b.to(device), bundles.to(device))
        # loss = loss_func(modelout, batch_size=loader.batch_size)
        #Youshu
        #预训练0.025,无ubloss
        #无预训练0.04,0.06
        #mixup 0.032,0.046
        #Repmixup  0.04 0.046
        #dm 0.05,0.058
        #loss = bpr_loss + 0.04*c_loss+graph_loss+0.046*ub_loss
        loss = bpr_loss + 0.04*c_loss+0.046*ub_loss
        #loss = bpr_loss

        #NetEase
        #0.091,0.03
        #loss = bpr_loss + 0.091* c_loss+0.03*ub_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % log_interval == 0:
            print('U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                100. * (i+1) / len(loader), loss))
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss

