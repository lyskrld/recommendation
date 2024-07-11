import pickle
import random
import numpy as np
from config import CONFIG


def E_score1(a, b):
    return np.sum(a * b) / (np.sqrt(
        np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))


def E_score2(a, b):
    return np.sum(np.power(a - b, 2))


# User-based Balanced Partition
def data_partition_1(train_bundle_items, train_user_items, train_user_bundles, k, T):
    ui_data = {}
    bi_data = {}
    ub_data = {}
    exist_user = []
    # 使用集合来存储已经提取过的元素
    seen = set()
    # 使用列表推导式生成结果列表
    for key, value in train_user_bundles:
        if key not in seen:
            seen.add(key)
            exist_user.append(key)
    print("exist user")
    print(len(exist_user))
    print(exist_user[:10])

    for key, value in train_user_items:
        if key not in ui_data:
            ui_data[key] = []
        ui_data[key].append(value)
    for key, value in train_user_bundles:
        if key not in ub_data:
            ub_data[key] = []
        ub_data[key].append(value)
    for key, value in train_bundle_items:
        if key not in bi_data:
            bi_data[key] = []
        bi_data[key].append(value)
    with open(CONFIG['path'] + '/' + CONFIG['dataset_name'] + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f, encoding='latin')
    uidW = uidW.detach().numpy()
    print("uidW")
    print(len(uidW))

    # Randomly select k centroids
    max_data = 1.2 * len(exist_user) / k
    centroids = random.sample(exist_user, k)
    print("centroids")
    print(centroids)
    # centro emb
    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i]]  # user_embedding
        centroembs.append(temp_u)

    for iteration in range(T):
        print("Iteration {}".format(iteration))
        UI = [{} for i in range(k)]  # {user:items}
        BI = [{} for i in range(k)]  # {bundle:items}
        UB = [{} for i in range(k)]  # {user:bundles}
        Scores = {}
        for i in exist_user:  # index of u
            for j in range(k):  # submodel index
                score_u = E_score2(uidW[i], centroembs[j])
                Scores[i, j] = -score_u  # {[uid,j]:scores} key:uid value:scores

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:  # uid not in fl
                if len(UB[Scores[i][0][1]]) < max_data:  # submodel capacity<maxdata
                    UB[Scores[i][0][1]][Scores[i][0][0]] = ub_data[Scores[i][0][0]]  # UB[submodel index][uid]=bundles
                    UI[Scores[i][0][1]][Scores[i][0][0]] = ui_data[Scores[i][0][0]]  # UI[submodel index][uid]=items
                    fl.add(Scores[i][0][0])  # add uid

        fl = set()
        for i in range(k):
            for j in UB[i].keys():
                for l in UB[i][j]:
                    if l not in fl:
                        BI[i][l] = bi_data[l]

        centroembs_next = []
        for i in range(k):
            temp_u = []
            for u in UB[i].keys():
                temp_u.append(uidW[u])
            centroembs_next.append(np.mean(temp_u))

        loss = 0.0
        length= []
        print("length of UB")
        for i in range(k):
            length.append(len(UB[i]))
        print(length)
        print("length of UI")
        length=[]
        for i in range(k):
            length.append(len(UI[i]))
        print(length)
        print("length of BI")
        length=[]
        for i in range(k):
            length.append(len(BI[i]))
        print(length)
        for i in range(k):
            score_u = E_score2(centroembs_next[i], centroembs[i])
            loss += score_u

        centroembs = centroembs_next
        print(iteration, loss)

    ub_shards = [[] for i in range(k)]
    ui_shards = [[] for i in range(k)]
    bi_shards = [[] for i in range(k)]
    for i in range(k):  # for each subdataset
        ub_shards[i] = []
        for j in UB[i].keys():  # for each user in ith subdataset
            for l in UB[i][j]:  # for each bundle in ith subdataset jth user
                ub_shards[i].append((j, l))

    for i in range(k):  # for each subdataset
        ui_shards[i] = []
        for j in UI[i].keys():  # for each user in ith subdataset
            for l in UI[i][j]:  # for each item in ith subdataset jth user
                ui_shards[i].append((j, l))

    for i in range(k):  # for each subdataset
        bi_shards[i] = []
        for j in BI[i].keys():  # for each bundle in ith subdataset
            for l in BI[i][j]:  # for each item in ith subdataset jth bundle
                bi_shards[i].append((j, l))

    return ub_shards, ui_shards, bi_shards
