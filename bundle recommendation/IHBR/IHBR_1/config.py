#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Youshu
CONFIG = {
    'name': 'TLGCN',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'IHGCN',
    'dataset_name': 'Youshu',
    'task': 'tune', 
    'eval_task': 'test',

    ## optimal hyperparameters
    #无预训练1e-3
    #预训练2e-4
    'lrs': [1e-3], 
    #无预训练0.3
    #预训练0.28
    'message_dropouts': [0.3],
    'node_dropouts': [0],
    'decays': [1e-4],

    ## hard negative sample and further train
    'sample': 'simple',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.3, 0.3], # probability 0.8
    'conti_train': 'log/Youshu/',
    'c_lambda': [0.04],
    'c_temps': [0.25],

    ## other settings
    'epochs': 500,
    'early': 10,
    'log_interval': 20,
    'test_interval': 5,
    'retry': 1,

    ## test path
    'test':['log/Youshu']
}

#NetEase
# CONFIG = {
#     'name': 'TLGCN',
#     'path': './data',
#     'log': './log',
#     'visual': './visual',
#     'gpu_id': "0",
#     'note': 'some_note',
#     'model': 'IHGCN',
#     'dataset_name': 'NetEase',
#     'task': 'tune',
#     'eval_task': 'test',

#     ## optimal hyperparameters
#     'lrs': [1e-3],
#     'message_dropouts': [0.3],
#     'node_dropouts': [0],
#     'decays': [1e-4],

#     ## hard negative sample and further train
#     'sample': 'simple',
#     'hard_window': [0.7, 1.0], # top 30%
#     'hard_prob': [0.3, 0.3], # probability 0.8

#     'conti_train': 'log/NetEase/',
#      'c_lambda': [0.1],
#     'c_temps': [0.25],

#     ## other settings
#     'epochs': 100,
#     'early': 5,
#     'log_interval': 20,
#     'test_interval': 5,
#     'retry': 1,

#     ## test path
#     'test':['log/NetEase']
# }