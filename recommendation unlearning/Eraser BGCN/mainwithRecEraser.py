#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
import setproctitle
from model import RecEraser_BGCN, RecEraser_BGCN_Info
from utils import check_overfitting, early_stop, logger
from metric import Recall, NDCG, MRR
from config import CONFIG
from train import train, train_local
from test import test, test_local
import loss
from itertools import product
import time
from time import time as Time
from tensorboardX import SummaryWriter
from data_partition import *
import datasetwithpart as dataset


def main():
    #  set env
    setproctitle.setproctitle(f"train{CONFIG['name']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cpu')

    #  fix seed
    seed = 123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    #  load data
    # data partition
    data_generator = dataset.Data(path=CONFIG['path'], name=CONFIG['dataset_name'], part_type=CONFIG['part_type'],
                                  part_num=CONFIG['part_num'], part_T=CONFIG['part_T'], task=CONFIG['task'])
    #  pretrain
    if 'pretrain' in CONFIG:
        pretrain = torch.load(CONFIG['pretrain'], map_location='cpu')
        print('load pretrain')

    #  local graph
    bundle_train_data_local= []
    bundle_test_data_local = []
    item_data_local = []
    assist_data_local = []
    graph = []
    for i in range(data_generator.part_num):
        bundle_train_data, bundle_test_data, item_data, assist_data = \
            dataset.get_dataset_local(CONFIG['path'], CONFIG['dataset_name'], data_generator, i, task=CONFIG['task'])
        bundle_train_data_local.append(bundle_train_data)
        bundle_test_data_local.append(bundle_test_data)
        item_data_local.append(item_data)
        assist_data_local.append(assist_data)
        print(len(bundle_train_data.U_B_pairs))
        C_UB_graph = bundle_train_data.ground_truth_u_b
        C_UI_graph = item_data.ground_truth_u_i
        C_BI_graph = assist_data.ground_truth_b_i
        graph.append([C_UB_graph, C_UI_graph, C_BI_graph])
    #  metric
    metrics = [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
    TARGET = 'Recall@20'

    #  loss
    loss_func = loss.BPRLoss('mean')

    #  log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'],
        f"{CONFIG['model']}_{CONFIG['task']}", ''), 'best', checkpoint_target=TARGET)

    theta = 0.6

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))

    for lr, decay, message_dropout, node_dropout \
            in product(CONFIG['lrs'], CONFIG['decays'], CONFIG['message_dropouts'], CONFIG['node_dropouts']):

        visual_path = os.path.join(CONFIG['visual'],
                                   CONFIG['dataset_name'],
                                   f"{CONFIG['model']}_{CONFIG['task']}",
                                   f"{time_path}@{CONFIG['note']}",
                                   f"lr{lr}_decay{decay}_medr{message_dropout}_nodr{node_dropout}")

        # model
        if CONFIG['model'] == 'RecEraser_BGCN':
            info = RecEraser_BGCN_Info(64, decay, message_dropout, node_dropout, 2)
            model = RecEraser_BGCN(info, data_generator, device, graph, pretrain=None).to(device)
        assert model.__class__.__name__ == CONFIG['model']

        # op
        op = optim.Adam(model.parameters(), lr=lr)
        # env
        env = {'lr': lr,
               'op': str(op).split(' ')[0],  # Adam
               'dataset': CONFIG['dataset_name'],
               'model': CONFIG['model'],
               'sample': CONFIG['sample'],
               }

        #  continue training
        if CONFIG['sample'] == 'hard' and 'conti_train' in CONFIG:
            model.load_state_dict(torch.load(CONFIG['conti_train']))
            print('load model and continue training')

        retry = CONFIG['retry']  # =1
        while retry >= 0:
            # log
            log.update_modelinfo(info, env, metrics)
            # train & test single model
            early = CONFIG['early']
            train_writer = SummaryWriter(log_dir=visual_path, comment='train')
            test_writer = SummaryWriter(log_dir=visual_path, comment='test')
            for local_num in range(data_generator.part_num):
                start = Time()
                print("this number {}".format(local_num))
                train_loader = DataLoader(bundle_train_data_local[local_num], 1024, True,
                                          num_workers=0, pin_memory=True)
                test_loader = DataLoader(bundle_test_data_local[local_num], 2048, False,
                                         num_workers=0, pin_memory=True)
                print("begin train local model{}".format(local_num))
                for epoch in range(CONFIG['epochs']):
                    # train_local
                    trainloss = train_local(model, epoch + 1, train_loader, op, device, CONFIG, loss_func, local_num)
                    train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)

                    # test_local
                    if epoch % CONFIG['test_interval'] == 0:
                        output_metrics = test_local(model, test_loader, device, CONFIG, metrics, local_num)

                        for metric in output_metrics:
                            test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric}, epoch)
                            if metric == output_metrics[0]:
                                test_writer.add_scalars('metric/single', {metric.get_title(): metric.metric}, epoch)

                        # log
                        log.update_log(metrics, model)

                        # check overfitting
                        if epoch > 10:
                            if check_overfitting(log.metrics_log, TARGET, 1, show=False):
                                break
                        # early stop
                        early = early_stop(
                            log.metrics_log[TARGET], early, threshold=0)
                        if early <= 0:
                            print("early stop")
                            break
                print('Train Local: {}: time = {:d}s'.format(local_num, int(Time() - start)))
            #save local model
            # train & test model_agg
            early = CONFIG['early']
            train_writer = SummaryWriter(log_dir=visual_path, comment='train')
            test_writer = SummaryWriter(log_dir=visual_path, comment='test')
            start = Time()
            bundle_train_data, bundle_test_data, item_data, assist_data = \
                data_generator.get_dataset()
            train_loader = DataLoader(bundle_train_data, 2048, True,
                                      num_workers=0, pin_memory=True)
            test_loader = DataLoader(bundle_test_data, 4096, False,
                                     num_workers=0, pin_memory=True)
            print("Begin train agg model")
            for epoch in range(CONFIG['epochs']):
                # train
                trainloss = train(model, epoch + 1, train_loader, op, device, CONFIG, loss_func)
                train_writer.add_scalars('loss/single', {"loss": trainloss}, epoch)

                # test
                if epoch % CONFIG['test_interval'] == 0:
                    output_metrics = test(model, test_loader, device, CONFIG, metrics)

                    for metric in output_metrics:
                        test_writer.add_scalars('metric/all', {metric.get_title(): metric.metric},
                                                epoch)
                        if metric == output_metrics[0]:
                            test_writer.add_scalars('metric/single',
                                                    {metric.get_title(): metric.metric}, epoch)

                    # log
                    log.update_log(metrics, model)

                    # check overfitting
                    if epoch > 10:
                        if check_overfitting(log.metrics_log, TARGET, 1, show=False):
                            break
                    # early stop
                    early = early_stop(
                        log.metrics_log[TARGET], early, threshold=0)
                    if early <= 0:
                        break
            print('Train Agg: time = {:d}s'.format(int(Time() - start)))
            train_writer.close()
            test_writer.close()

            log.close_log(TARGET)
            retry = -1
    log.close()


if __name__ == "__main__":
    main()
