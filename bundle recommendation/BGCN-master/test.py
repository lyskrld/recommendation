#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os

def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate() 
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))  
            pred_b -= 1e8*train_mask_u_b.to(device)
            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
    print('Test: time={:d}s'.format(int(time()-start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    print("users_feature")
    print(rs[0][1][0:3])
    print("bundles_feature")
    print(rs[1][1][0:3])
    return metrics

def test_local(model, loader, device, CONFIG, metrics, local_num):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.local_propagate(0,local_num)
        for users, ground_truth_u_b, train_mask_u_b in loader:
            pred_b = model.evaluate(rs, users.to(device))
            pred_b -= 1e8*train_mask_u_b.to(device)
            for metric in metrics:
                metric(pred_b, ground_truth_u_b.to(device))
    print('Test Local: {} Test: time={:d}s'.format(local_num, int(time()-start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    print("users_feature")
    print(rs[0][1][0:3])
    print("bundles_feature")
    print(rs[1][1][0:3])
    return metrics

