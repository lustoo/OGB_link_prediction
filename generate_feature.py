import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm
import pickle


def resource_allocation(adj_matrix, link_list, batch_size=32768, cate_index=None, cate_degree=None):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    A = adj_matrix
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 1
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()

    link_index = link_list.t()  # (2,:)
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    ra = []
    cn = []
    aa = []

    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        ra.append(np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten())
        aa.append(np.array(np.sum(A[src].multiply(D_log[dst]), 1)).flatten())
        cn.append(np.array(np.sum(A[src].multiply(D_common[dst]), 1)).flatten())
        # break

    cn = np.concatenate(cn, 0)
    ra = np.concatenate(ra, 0)
    aa = np.concatenate(aa, 0)
    return torch.FloatTensor(cn), torch.FloatTensor(ra), torch.FloatTensor(aa)


def edge2feature(A, edges, save_name, one_hot):
    batch_size = 1024
    model_predictor = resource_allocation
    #############
    cn, ra, aa = model_predictor(A, edges, batch_size=batch_size)  # 分数

    features = torch.stack((cn, ra, aa), dim=1)  # (n,3)
    # one hot
    features = torch.cat((one_hot[edges[:, 0], :], one_hot[edges[:, 1], :], features), dim=1)

    file = open(save_name, 'wb')
    pickle.dump(features, file)


def main():
    dataset_name = 'ogbl-ppa'
    dataset = PygLinkPropPredDataset(name=dataset_name)
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    # graph_construct
    print("Constructing graph.")
    train_edges_raw = np.array(split_edge['train']['edge'])
    train_edges_reverse = np.array(
        [train_edges_raw[:, 1], train_edges_raw[:, 0]]).transpose()
    train_edges = np.concatenate(
        [train_edges_raw, train_edges_reverse], axis=0)
    edge_weight = torch.ones(train_edges.shape[0], dtype=int)
    A = ssp.csr_matrix(
        (edge_weight, (train_edges[:, 0], train_edges[:, 1])), shape=(
            data.num_nodes, data.num_nodes)
    )

    # train pos edges
    pos_train_edge = split_edge['train']['edge']
    edge2feature(A, pos_train_edge[:10000000, :], 'f_train_pos_edge_1.pickle', data.x)
    edge2feature(A, pos_train_edge[10000000:, :], 'f_train_pos_edge_1.pickle', data.x)
    # random neg edges
    random.seed(1)
    torch.manual_seed(1)
    random_negative_edges = torch.randint(0, data.num_nodes, pos_train_edge.size(), dtype=torch.long)
    edge2feature(A, random_negative_edges[:10000000, :], 'f_random_neg_edge_1.pickle', data.x)
    edge2feature(A, random_negative_edges[10000000:, :], 'f_random_neg_edge_2.pickle', data.x)
    # val
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    edge2feature(A, pos_valid_edge, 'f_pos_valid_edge.pickle', data.x)
    edge2feature(A, neg_valid_edge, 'f_neg_valid_edge.pickle', data.x)

    # test
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']
    edge2feature(A, pos_test_edge, 'f_pos_test_edge.pickle', data.x)
    edge2feature(A, neg_test_edge, 'f_neg_test_edge.pickle', data.x)



def load(name):
    features = pickle.load(name)
    pass


main()
