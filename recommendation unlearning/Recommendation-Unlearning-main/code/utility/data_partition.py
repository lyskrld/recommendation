from utility.parser import parse_args
import pickle
import copy
import random
import numpy as np
args = parse_args()


def E_score1(a,b):
    return np.sum(a * b) / (np.sqrt(
                    np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))) )

def E_score2(a,b):
    return np.sum(np.power(a-b, 2))


# Interaction-based Balanced Partition
def data_partition_1(train_items,k,T):

    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f,encoding='latin')
    with open(args.data_path + args.dataset + '/item_pretrain.pk', 'rb') as f:
        iidW = pickle.load(f,encoding='latin')

    # get_data_interactions_1
    data = []
    for i in train_items:
        for j in train_items[i]:
            data.append([i, j])

    # Randomly select k centroids
    max_data = 1.2 * len(data) / k
    centroids = random.sample(data, k)


    # centro emb

    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i][0]]#user_embedding
        temp_i = iidW[centroids[i][1]]#item)embedding
        centroembs.append([temp_u, temp_i])


    for _ in range(T):
        C = [{} for i in range(k)]
        C_num=[0 for i in range(k)]
        Scores = {}
        for i in range(len(data)):#index of(u,i)
            for j in range(k):#submodel index

                score_u = E_score2(uidW[data[i][0]],centroembs[j][0])
                score_i = E_score2(iidW[data[i][1]],centroembs[j][1])

                
                Scores[i, j] = -score_u * score_i#{((u,i),j):scores}


        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)
        print("data[0]")
        print(data[0])
        print("Scores[0]")
        print(Scores[0])
        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:#data_id not in fl

                if C_num[Scores[i][0][1]] < max_data:#submodel num<maxdata
                    if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:#user not in C[submodel_index]
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]]=[data[Scores[i][0][0]][1]]#C[submodel_index][user]=[item]
                    else:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])#C[submodel_index][user].append(item)
                    fl.add(Scores[i][0][0])#add data_id
                    C_num[Scores[i][0][1]] +=1#submodel capacity++

        centroembs_next = []
        for i in range(k):
            temp_u = []
            temp_i = []

            for j in C[i].keys():#user
                for l in C[i][j]:#item
                    temp_u.append(uidW[j])#
                    temp_i.append(iidW[l])
            centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])#mean anchor

        loss = 0.0

        for i in range(k):
            score_u = E_score2(centroembs_next[i][0],centroembs[i][0])

            score_i = E_score2(centroembs_next[i][1],centroembs[i][1])

            loss += (score_u * score_i)

        centroembs = centroembs_next
        for i in range(k):
            print(C_num[i])#submodel capacity

        print(_, loss)

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]

    for i in range(k):#for each subdataset
        users[i] = list(C[i].keys())#C[i].keys():users in ith subdataset       users[i]:users in ith subdataset
        for j in C[i].keys():#for each user in ith subdataset
            for l in C[i][j]:#for each item in ith subdataset jth user
                if l not in items[i]:#if item not in ith subdataset     items[i]:items in ith subdataset
                    items[i].append(l)


    return C,users,items#C:all subdatasets, all users, all items


# User-based Balanced Partition
def data_partition_2(train_items,k,T):
    data = train_items

    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'r') as f:
        uidW = pickle.load(f)

    # Randomly select k centroids
    max_data = 1.2 * len(data) / k
    # print data
    centroids = random.sample(data.keys(), k)

    # centro emb
    # print centroids
    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i]]
        centroembs.append(temp_u)

    for _ in range(T):
        C = [{} for i in range(k)]
        Scores = {}
        for i in data.keys():
            for j in range(k):
                score_u = E_score2(uidW[i],centroembs[j])

                Scores[i, j] = -score_u

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:
                if len(C[Scores[i][0][1]]) < max_data:
                    C[Scores[i][0][1]][Scores[i][0][0]] = data[Scores[i][0][0]]
                    fl.add(Scores[i][0][0])

        centroembs_next = []
        for i in range(k):
            temp_u = []
            for u in C[i].keys():
                temp_u.append(uidW[u])
            centroembs_next.append(np.mean(temp_u))

        loss = 0.0

        for i in range(k):
            print(len(C[i]))

        for i in range(k):
            score_u = E_score2(centroembs_next[i],centroembs[i])
            loss += score_u

        centroembs = centroembs_next
        print(_, loss)

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        users[i]=C[i].keys()
        for j in C[i].keys():
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)

    return C,users,items


#Random
def data_partition_3(train_items,k):

    data = []
    for i in train_items:
        for j in train_items[i]:
            data.append([i, j])

    index = list(range(len(data)))

    random.shuffle(index) 

    elem_num =len(data) / k

    C = [{} for i in range(k)]


    for idx in range(k):
        start = idx*elem_num
        if idx!= k-1:
            end = (idx+1)*elem_num
        else:
            end = len(data)
        for i in index[start:end]:
            if data[i][0] not in C[idx]:
                C[idx][data[i][0]]=[data[i][1]]
            else:
                C[idx][data[i][0]].append(data[i][1])

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        users[i]=C[i].keys()
        for j in C[i].keys():
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)

    return C,users,items
















