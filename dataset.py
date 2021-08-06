import torch.utils.data as data
import pickle
import torch


class TrainSet(data.Dataset):
    def __init__(self):
        self.one = torch.ones(1, dtype=torch.long)
        self.zero = torch.zeros(1, dtype=torch.long)
        f1 = open('f_train_pos_edge_1.pickle', 'rb')
        f2 = open('f_train_pos_edge_2.pickle', 'rb')
        pos_feature = torch.cat((pickle.load(f1), pickle.load(f2)), dim=0)
        f1.close()
        f2.close()
        f1 = open('f_random_neg_edge_1.pickle', 'rb')
        f2 = open('f_random_neg_edge_2.pickle', 'rb')
        neg_feature = torch.cat((pickle.load(f1), pickle.load(f2)), dim=0)
        f1.close()
        f2.close()
        self.neg_n = neg_feature.shape[0]
        self.pos_n = pos_feature.shape[0]
        self.feature_pos = pos_feature
        self.feature_neg = neg_feature



    def __getitem__(self, item):

        if item < self.pos_n:
            return self.feature_pos[item], self.one

        else:
            return self.feature_neg[item - self.pos_n], self.zero

    def __len__(self):
        return self.pos_n + self.neg_n


class TestSet(data.Dataset):
    def __init__(self, phase, pn):
        self.one = torch.ones(1)
        self.zero = torch.zeros(1)
        self.pn = pn
        self.phase = phase
        with open('f_{}_{}_edge.pickle'.format(pn, phase), 'rb') as file:
            self.feature = pickle.load(file)
            self.length = self.feature.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.pn == 'pos':
            return self.feature[item], self.one
        else:
            return self.feature[item], self.zero
