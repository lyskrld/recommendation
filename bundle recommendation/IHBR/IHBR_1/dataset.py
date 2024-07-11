#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scipy.sparse as sp 
from torch.utils.data import Dataset
from config import CONFIG
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import enum
import pickle



def sparse_ones(indices, size, dtype=torch.float):
    one = torch.ones(indices.shape[1], dtype=dtype)
    return torch.sparse.FloatTensor(indices, one, size=size).to(dtype)

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BasicDataset(Dataset):
    '''
    generate dataset from raw *.txt
    contains:
        tensors like (`user`, `bundle_p`, `bundle_n1`, `bundle_n2`, ...) for BPR (use `self.user_bundles`)
    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `neg_sample`: the number of negative samples for each user-bundle_p pair
    - `seed`: seed of `np.random`
    '''

    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_bundles, self.num_items  = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(self.name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]
    def load_U_B_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(self.task)), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_B_I_affiliation(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))


class BundleTrainDataset(BasicDataset):
    def __init__(self, path, name, item_data, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(self.ground_truth_u_b, 'U-B statistics in train')

        if CONFIG['sample'] == 'hard': 
            #  1. u_p --> b_n1
            u_b_from_i = item_data.ground_truth_u_i @ assist_data.ground_truth_b_i.T
            u_b_from_i = u_b_from_i.todense()
            bn1_window = [int(i*self.num_bundles) for i in CONFIG['hard_window']]
            self.u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:,bn1_window[0]:bn1_window[1]]

            #  2. b_p --> b_n2
            overlap_graph = assist_data.ground_truth_b_i @ assist_data.ground_truth_b_i.T
            overlap_graph = overlap_graph.todense()
            bn2_window = [int(i*self.num_bundles) for i in CONFIG['hard_window']]
            self.b_b_for_neg_sample = np.argsort(overlap_graph, axis=1)[:,bn2_window[0]:bn2_window[1]]

    def __getitem__(self, index):
        user_b, pos_bundle = self.U_B_pairs[index]
        all_bundles = [pos_bundle]
        if CONFIG['sample'] == 'simple':
            while True:
                i = np.random.randint(self.num_bundles)
                if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                    all_bundles.append(i)
                    if len(all_bundles) == self.neg_sample+1:
                        break
        elif CONFIG['sample'] == 'hard':
            hard_probability = round(np.random.uniform(0, 1), 1)
            if  hard_probability <= CONFIG['hard_prob'][0]:
                while True:
                    i = np.random.randint(self.u_b_for_neg_sample.shape[1])
                    b_n1 = self.u_b_for_neg_sample[user_b, i]
                    if self.ground_truth_u_b[user_b, b_n1] == 0 and not b_n1 in all_bundles:
                        all_bundles.append(b_n1)
                        if len(all_bundles) == self.neg_sample+1:
                            break
            elif CONFIG['hard_prob'][0] < hard_probability \
                <= CONFIG['hard_prob'][0] + CONFIG['hard_prob'][1]:
                while True:
                    i = np.random.randint(self.b_b_for_neg_sample.shape[1])
                    b_n2 = self.b_b_for_neg_sample[pos_bundle, i]
                    if self.ground_truth_u_b[user_b, b_n2] == 0 and not b_n2 in all_bundles:
                        all_bundles.append(b_n2)
                        if len(all_bundles) == self.neg_sample+1:
                            break
            else:
                while True:
                    i = np.random.randint(self.num_bundles)
                    if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                        all_bundles.append(i)
                        if len(all_bundles) == self.neg_sample+1:
                            break
        else:
            raise ValueError(r"sample's method is wrong")

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.U_B_pairs)  


class BundleTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task='test'):
        super().__init__(path, name, task, None)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(self.ground_truth_u_b, 'U-B statistics in test')

        self.train_mask_u_b = train_dataset.ground_truth_u_b
        self.users = torch.arange(self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(self.num_bundles, dtype=torch.long)
        assert self.train_mask_u_b.shape == self.ground_truth_u_b.shape

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth_u_b[index].toarray()).squeeze(),  \
            torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze(),  \

    def __len__(self):
        return self.ground_truth_u_b.shape[0]

class ItemDataset(BasicDataset):
    def __init__(self, path, name, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-I
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(self.ground_truth_u_i, 'U-I statistics')

    def __getitem__(self, index):
        user_i, pos_item = self.U_I_pairs[index]
        all_items = [pos_item]
        while True:
            j = np.random.randint(self.num_items)
            if self.ground_truth_u_i[user_i, j] == 0 and not j in all_items:
                all_items.append(j)
                if len(all_items) == self.neg_sample+1:
                    break

        return torch.LongTensor([user_i]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.U_I_pairs)  


class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # B-I
        self.B_I_pairs = self.load_B_I_affiliation()
        indice = np.array(self.B_I_pairs, dtype=np.int32)
        values = np.ones(len(self.B_I_pairs), dtype=np.float32)
        self.ground_truth_b_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(self.ground_truth_b_i, 'B-I statistics')


def get_dataset(path, name, task='tune', seed=123):
    assist_data = AssistDataset(path, name)
    print('finish loading assist data')
    item_data = ItemDataset(path, name, assist_data, seed=seed)
    print('finish loading item data')

    bundle_train_data = BundleTrainDataset(path, name, item_data, assist_data, seed=seed)
    print('finish loading bundle train data')
    bundle_test_data = BundleTestDataset(path, name, bundle_train_data, task=task)
    print('finish loading bundle test data')

    return bundle_train_data, bundle_test_data, item_data, assist_data

class MyDataset:
    """
    The dataset class, supports all the datasets (according to the dataset_string argument).
    Contain user-bundle, user-item, and item-bundle related properties.
    The class also supports graph sampling if use_graph_sampling is passed, but it did not improve our results and so was not used.
    """
    def __init__(self, path, use_mini_test=False):
        use_graph_sampling_cache = True
        self.path = path
        self.dataset_name = CONFIG['dataset_name']
        self.num_users, self.num_bundles, self.num_items  = self.__load_data_size()

        self.initiate_bundle_items_map()
        self.initiate_user_item_properties()
        self.initiate_train_user_bundle_properties(use_graph_sampling_cache)
        self.initiate_test_user_bundle_properties(use_mini_test)
        self.initiate_tune_user_bundle_properties()
        if False:
            if use_graph_sampling_cache:
                self.items_train_triplets = self.load_data_triplets(product_type=Product.item)
                self.bundle_train_triplets = self.load_data_triplets(product_type=Product.bundle)
                self.num_user_item_triplets = len(self.items_train_triplets)
                self.num_user_bundle_triplets = len(self.bundle_train_triplets)
            else:
                self.items_train_triplets = self.create_data_triplets(product_type=Product.item)
                self.bundle_train_triplets = self.create_data_triplets(product_type=Product.bundle)
                self.num_user_item_triplets = len(self.items_train_triplets)
                self.num_user_bundle_triplets = len(self.bundle_train_triplets)
                self.triplets_to_csv(self.items_train_triplets, Product.item)
                self.triplets_to_csv(self.bundle_train_triplets, Product.bundle)

    

    def __load_data_size(self):
        with open(os.path.join(self.path, self.dataset_name, f'{self.dataset_name}_data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def load_U_B_interaction(self, task, use_mini=False):
        file_name = f"user_bundle_{task}-minimini.txt" if use_mini else f"user_bundle_{task}.txt"
        with open(os.path.join(self.path, self.dataset_name, file_name), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.dataset_name, 'user_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

    def load_B_I_interaction(self):
        with open(os.path.join(self.path, self.dataset_name, 'bundle_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))


    def triplets_to_csv(self, triplets, product_type):
        with open(os.path.join(self.path, self.dataset_name, f'{str(product_type)}_train_triplets.txt'), 'w') as f:
            f.writelines([f"{','.join([str(i) for i in t])}\n" for t in triplets])

    def pairs_list_to_ground_truth_mtx(self, pairs_list, num_users, num_products):
        indice = np.array(pairs_list, dtype=np.int32)
        values = np.ones(len(pairs_list), dtype=np.float32)
        return sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(num_users, num_products)).tocsr()

    def initiate_train_user_bundle_properties(self, using_triplets_cache=True):
        self.U_B_pairs_train = self.load_U_B_interaction("train")
        self.ground_truth_u_b_train = self.pairs_list_to_ground_truth_mtx(self.U_B_pairs_train, self.num_users, self.num_bundles)
        if not using_triplets_cache:
            df = pd.read_csv(os.path.join(self.path, self.dataset_name, f'user_bundle_train.txt'), delimiter="\t", names=["user", "product"])
            self.bundle_interaction_count = df["product"].value_counts()
            self.bundle_sorted_by_degree = list(self.bundle_interaction_count.index)
            self.num_bundles_with_an_interaction = len(self.bundle_sorted_by_degree)
            self.bundle_to_index = dict([(p, i) for i, p in enumerate(self.bundle_sorted_by_degree)])

    def initiate_test_user_bundle_properties(self, use_mini_test=False):
        U_B_pairs_test = self.load_U_B_interaction("test", use_mini_test)
        U_I_pairs = self.load_U_I_interaction()
        self.ground_truth_u_b_all_test = self.pairs_list_to_ground_truth_mtx(U_B_pairs_test, self.num_users, self.num_bundles)
        self.ground_truth_u_i_all_test = self.pairs_list_to_ground_truth_mtx(U_I_pairs, self.num_users, self.num_items)
        self.test_relevant_users = list(set([pair[0] for pair in U_B_pairs_test]))
        self.ground_truth_u_b_test_relevant_users = self.ground_truth_u_b_all_test[self.test_relevant_users]
        self.train_mask_only_relevant_test = self.ground_truth_u_b_train[self.test_relevant_users]

    def initiate_tune_user_bundle_properties(self):
        U_B_pairs_tune = self.load_U_B_interaction("tune")
        self.ground_truth_u_b_all_tune = self.pairs_list_to_ground_truth_mtx(U_B_pairs_tune, self.num_users, self.num_bundles)
        self.tune_relevant_users = list(set([pair[0] for pair in U_B_pairs_tune]))
        self.ground_truth_u_b_tune_relevant_users = self.ground_truth_u_b_all_tune[self.tune_relevant_users]
        self.train_mask_only_relevant_tune = self.ground_truth_u_b_train[self.tune_relevant_users]

    def initiate_bundle_items_map(self):
        self.B_I_pairs = self.load_B_I_interaction()
        self.ground_truth_b_i = self.pairs_list_to_ground_truth_mtx(self.B_I_pairs, self.num_bundles, self.num_items)

    def initiate_user_item_properties(self):
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        input_file_name = os.path.join(self.path, self.dataset_name, 'user_item.txt')
        df = pd.read_csv(input_file_name, delimiter="\t", names=["user", "product"])
        self.items_interaction_count = df["product"].value_counts()
        self.items_sorted_by_degree = list(self.items_interaction_count.index)
        self.num_items_with_an_interaction = len(self.items_sorted_by_degree)
        self.item_to_index = dict([(p, i) for i, p in enumerate(self.items_sorted_by_degree)])

    

    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
