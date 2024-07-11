#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch.nn import init

from .new_model_base import Info, Model
from config import CONFIG
import pickle
import os
import datasetwithpart as dataset

def graph_generating(raw_graph, row, col):
    if raw_graph.shape == (row, col):
        graph = sp.bmat([[sp.identity(raw_graph.shape[0]), raw_graph],
                         [raw_graph.T, sp.identity(raw_graph.shape[1])]])
    else:
        raise ValueError(r"raw_graph's shape is wrong")
    return graph


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    indices = list(indices)
    values = list(values)
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph


class RecEraser_BGCN_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers


class RecEraser_BGCN(Model):
    def get_infotype(self):
        return RecEraser_BGCN_Info

    def __init__(self, info, generator, device, graph, pretrain=None):
        super().__init__(info, generator, create_embeddings=True)
        self.part_num=generator.part_num
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.part_num, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        self.epison = 1e-8
        self.attention_size=int(self.embedding_size / 2)
        self.atom_graph = []
        self.dnns_atom = []
        self.non_atom_graph = []
        self.dnns_non_atom = []
        self.pooling_graph = []
        self.atom_users_feature=[]
        self.non_atom_users_feature= []
        self.atom_bundles_feature= []
        self.non_atom_bundles_feature= []
        # copy from info
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.device = device
        self.weights=self._init_weights()
        #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)
        for local_num in range(self.part_num):
            assert isinstance(graph[local_num], list)
            ub_graph, ui_graph, bi_graph = graph[local_num]
            print("Graph {}".format(local_num))
            #  deal with weights
            bi_norm = sp.diags(1 / (np.sqrt((bi_graph.multiply(bi_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ bi_graph
            bb_graph = bi_norm @ bi_norm.T

            #  pooling graph
            bundle_size = bi_graph.sum(axis=1) + 1e-8
            bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph

            if ui_graph.shape == (self.num_users, self.num_items):
                # add self-loop
                atom_graph = sp.bmat([[sp.identity(ui_graph.shape[0]), ui_graph],
                                      [ui_graph.T, sp.identity(ui_graph.shape[1])]])
            else:
                raise ValueError(r"raw_graph's shape is wrong")
            self.atom_graph.append(to_tensor(laplace_transform(atom_graph)).to(device)) # 拉普拉斯变换
            print('finish generating atom graph')

            if ub_graph.shape == (self.num_users, self.num_bundles) \
                    and bb_graph.shape == (self.num_bundles, self.num_bundles):
                # add self-loop
                non_atom_graph = sp.bmat([[sp.identity(ub_graph.shape[0]), ub_graph],
                                          [ub_graph.T, bb_graph]])
            else:
                raise ValueError(r"raw_graph's shape is wrong")
            self.non_atom_graph.append(to_tensor(laplace_transform(non_atom_graph)).to(device))
            print('finish generating non-atom graph')

            self.pooling_graph.append(to_tensor(bi_graph).to(device))
            print('finish generating pooling graph')

            # Layers
            self.dnns_atom.append(nn.ModuleList([nn.Linear(
                self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)]))
            self.dnns_non_atom.append(nn.ModuleList([nn.Linear(
                self.embedding_size * (l + 1), self.embedding_size) for l in range(self.num_layers)]))

        # pretrain
        if not pretrain is None:
            self.users_feature.data = F.normalize(
                pretrain['users_feature'])
            self.items_feature.data = F.normalize(
                pretrain['items_feature'])
            self.bundles_feature.data = F.normalize(
                pretrain['bundles_feature'])

    def _init_weights(self):
        all_weights = dict()
        initializer = init.normal_

        # atom_users attention
        std_dev = torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.embedding_size)))
        all_weights['WA'] = nn.Parameter(torch.empty(self.embedding_size, self.attention_size,dtype=torch.float32))
        init.normal_(all_weights['WA'] , mean=0.0, std=std_dev.item())
        all_weights['BA'] = nn.Parameter(torch.zeros(self.attention_size), requires_grad=True)
        all_weights['HA'] = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01, requires_grad=True)

        # non_atom_users attention
        std_dev = torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.embedding_size)))
        all_weights['WB'] = nn.Parameter(torch.empty(self.embedding_size, self.attention_size,dtype=torch.float32))
        init.normal_(all_weights['WB'] , mean=0.0, std=std_dev.item())
        all_weights['BB'] = nn.Parameter(torch.zeros(self.attention_size), requires_grad=True)
        all_weights['HB'] = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01, requires_grad=True)

        # atom_bundles attention
        std_dev = torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.embedding_size)))
        all_weights['WC'] = nn.Parameter(torch.empty(self.embedding_size, self.attention_size,dtype=torch.float32))
        init.normal_(all_weights['WC'] , mean=0.0, std=std_dev.item())
        all_weights['BC'] = nn.Parameter(torch.zeros(self.attention_size), requires_grad=True)
        all_weights['HC'] = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01, requires_grad=True)

        # non_atom_bundles attention
        std_dev = torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.embedding_size)))
        all_weights['WD'] = nn.Parameter(torch.empty(self.embedding_size, self.attention_size,dtype=torch.float32))
        init.normal_(all_weights['WD'] , mean=0.0, std=std_dev.item())
        all_weights['BD'] = nn.Parameter(torch.zeros(self.attention_size), requires_grad=True)
        all_weights['HD'] = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01, requires_grad=True)

        # trans weights
        weight_initializer = nn.init.normal_  # 或者使用 nn.init.uniform_
        weight = torch.empty(self.part_num, self.embedding_size, self.embedding_size)
        weight_initializer(weight, mean=0.0, std=0.01)
        all_weights['trans_W'] = nn.Parameter(weight, requires_grad=True)

        weight_initializer = nn.init.normal_  # 或者使用 nn.init.uniform_
        weight = torch.empty(self.part_num, self.embedding_size, self.embedding_size)
        weight_initializer(weight, mean=0.0, std=0.01)
        all_weights['trans_B'] = nn.Parameter(weight, requires_grad=True)

        return all_weights

    def one_propagate(self, graph, A_feature, B_feature, dnns):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(
            indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = self.mess_dropout(torch.cat([self.act(
                dnns[i](torch.matmul(graph, features))), features], 1))
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature

    def attention_based_agg(self, embs, flag):
        if flag == 'auf':#atom_users_feature
            embs_w = torch.exp(
                torch.matmul(F.relu(
                    torch.matmul(embs, self.weights['WA']) + self.weights['BA']),
                             self.weights['HA']))
            embs_w = embs_w / torch.sum(embs_w, dim=1, keepdim=True)

        if flag== 'nauf':#non_atom_users_feature
            embs_w = torch.exp(
                torch.matmul(F.relu(
                    torch.matmul(embs, self.weights['WB']) + self.weights['BB']),
                    self.weights['HB']))
            embs_w = embs_w / torch.sum(embs_w, dim=1, keepdim=True)

        if flag== 'abf':#atom_bundles_feature
            embs_w = torch.exp(
                torch.matmul(F.relu(
                    torch.matmul(embs, self.weights['WC']) + self.weights['BC']),
                    self.weights['HC']))
            embs_w = embs_w / torch.sum(embs_w, dim=1, keepdim=True)

        if flag== 'nabf':#non_atom_bundles_feature
            embs_w = torch.exp(
                torch.matmul(F.relu(
                    torch.matmul(embs, self.weights['WD']) + self.weights['BD']),
                    self.weights['HD']))
            embs_w = embs_w / torch.sum(embs_w, dim=1, keepdim=True)

        agg_emb = torch.sum(embs_w * embs, dim=1)

        return agg_emb, embs_w

    def propagate(self):
        #  =============================  item level propagation  =============================
        atom_users_feature, atom_items_feature = self.one_propagate(
            self.atom_graph, self.users_feature, self.items_feature, self.dnns_atom)
        atom_bundles_feature = F.normalize(torch.matmul(self.pooling_graph, atom_items_feature))
        #  ============================= bundle level propagation =============================
        non_atom_users_feature, non_atom_bundles_feature = self.one_propagate(
            self.non_atom_graph, self.users_feature, self.bundles_feature, self.dnns_non_atom)

        users_feature = [atom_users_feature, non_atom_users_feature]
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]
        return users_feature, bundles_feature

    def local_propagate(self, epoch, local_num):
        #  =============================  item level propagation  =============================
        atom_users_feature, atom_items_feature = self.one_propagate(
            self.atom_graph[local_num], self.users_feature[:, local_num], self.items_feature[:, local_num], self.dnns_atom[local_num])
        atom_bundles_feature = F.normalize(torch.matmul(self.pooling_graph[local_num], atom_items_feature))

        #  ============================= bundle level propagation =============================
        non_atom_users_feature, non_atom_bundles_feature = self.one_propagate(
            self.non_atom_graph[local_num], self.users_feature[:, local_num], self.bundles_feature[:, local_num], self.dnns_non_atom[local_num])
        if(epoch==10):
            self.atom_users_feature.append(atom_users_feature)
            self.atom_bundles_feature.append(atom_bundles_feature)
            self.non_atom_users_feature.append(non_atom_users_feature)
            self.non_atom_bundles_feature.append(non_atom_bundles_feature)
        users_feature = [atom_users_feature, non_atom_users_feature]
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]
        return users_feature, bundles_feature

    def agg_propagate(self):
        atom_users_feature=torch.cat(self.atom_users_feature, dim=0)
        atom_bundles_feature=torch.cat(self.atom_bundles_feature, dim=0)
        non_atom_users_feature=torch.cat(self.non_atom_users_feature, dim=0)
        non_atom_bundles_feature=torch.cat(self.non_atom_bundles_feature, dim=0)
        print("atom_users_feature.shape")
        print(atom_users_feature.shape)

        atom_users_feature = torch.matmul(atom_users_feature, self.weights['trans_W']) + self.weights['trans_B']
        print("atom_users_feature.shape")
        print(atom_users_feature.shape)
        atom_users_embedding, atom_users_weight = self.attention_based_agg(atom_users_feature, 'auf')
        print("atom_users_embedding.shape")
        print(atom_users_embedding.shape)
        non_atom_users_feature = torch.matmul(non_atom_users_feature, self.weights['trans_W']) + self.weights['trans_B']
        non_atom_users_embedding, non_atom_users_weight = self.attention_based_agg(non_atom_users_feature, 'nauf')

        atom_bundles_feature = torch.matmul(atom_bundles_feature, self.weights['trans_W']) + self.weights['trans_B']
        atom_bundles_embedding, atom_bundles_weight = self.attention_based_agg(atom_bundles_feature, 'abf')
        non_atom_bundles_feature = torch.matmul(non_atom_bundles_feature, self.weights['trans_W']) + self.weights['trans_B']
        non_atom_bundles_embedding, non_atom_bundles_weight = self.attention_based_agg(non_atom_bundles_feature, 'nabf')

        users_embedding=[atom_users_embedding,non_atom_users_embedding]
        bundles_embedding=[atom_bundles_embedding,non_atom_bundles_embedding]
        return users_embedding, bundles_embedding

    def save_feature(self, A_feature, B_feature):
        with open(CONFIG['path'] + '/' + CONFIG['dataset_name'] + '/user_pretrain.pk', 'wb') as f:
            pickle.dump(A_feature, f)
        with open(CONFIG['path'] + '/' + CONFIG['dataset_name'] + '/bundle_pretrain.pk', 'wb') as f:
            pickle.dump(B_feature, f)

    def predict(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature  # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        pred = torch.sum(users_feature_atom * bundles_feature_atom, 2) \
               + torch.sum(users_feature_non_atom * bundles_feature_non_atom, 2)
        return pred

    def forward(self, users, bundles, epoch, local_num):
        if local_num>=0:
            users_feature, bundles_feature = self.local_propagate(epoch, local_num)
        else:
            print("local_num")
            print(local_num)
            users_feature, bundles_feature = self.agg_propagate()
        users_embedding = [i[users].expand(- 1, bundles.shape[1], -1) for i in
                           users_feature]  # u_f --> batch_f --> batch_n_f
        bundles_embedding = [i[bundles] for i in bundles_feature]  # b_f --> batch_n_f
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_embedding, bundles_embedding)
        return pred, loss

    def regularize(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature  # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # batch_n_f
        loss = self.embed_L2_norm * \
               ((users_feature_atom ** 2).sum() + (bundles_feature_atom ** 2).sum() + \
                (users_feature_non_atom ** 2).sum() + (bundles_feature_non_atom ** 2).sum())
        return loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all bundles for `users` by `propagate_result`
        '''
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]  # batch_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature  # b_f
        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) \
                 + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())  # batch_b
        return scores

