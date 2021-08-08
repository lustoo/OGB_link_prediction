import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainSet, TestSet
from model import MLP
import torch.optim as optim
import os
import time
from utils import AverageMeter, get_opt
from ogb.linkproppred import Evaluator


def get_scores(model, data_loader, idx):
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            edges, labels = batch
            edges = edges.cuda()[:, idx]
            out = model(edges)
            scores.append(out[:, 1])
    return torch.cat(scores, 0)


def test(model, pos_valid_loader, neg_valid_loader, pos_test_loader, neg_test_loader, idx):
    model.eval()
    pos_valid_pred = get_scores(model, pos_valid_loader, idx)
    neg_valid_pred = get_scores(model, neg_valid_loader, idx)
    pos_test_pred = get_scores(model, pos_test_loader, idx)
    neg_test_pred = get_scores(model, neg_test_loader, idx)
    evaluator = Evaluator(name='ogbl-ppa')
    evaluator.K = 100

    valid_acc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['hits@100']

    test_acc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['hits@100']

    return valid_acc, test_acc


opt = get_opt()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

model = MLP(input_size=119 if opt.sim == 'all' else 117, class_num=2)
model = model.cuda()

model.train()
lossf = nn.CrossEntropyLoss()
train_set = TrainSet()
train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=8, pin_memory=True,
                          shuffle=True)  # 512 0

pos_valid_set = TestSet(pn='pos', phase='valid')
pos_valid_loader = DataLoader(dataset=pos_valid_set, batch_size=opt.batch_size, num_workers=8, pin_memory=True,
                              shuffle=False)

neg_valid_set = TestSet(pn='neg', phase='valid')
neg_valid_loader = DataLoader(dataset=neg_valid_set, batch_size=opt.batch_size, num_workers=8, pin_memory=True,
                              shuffle=False)

pos_test_set = TestSet(pn='pos', phase='test')
pos_test_loader = DataLoader(dataset=pos_test_set, batch_size=opt.batch_size, num_workers=8, pin_memory=True,
                             shuffle=False)

neg_test_set = TestSet(pn='neg', phase='test')
neg_test_loader = DataLoader(dataset=neg_test_set, batch_size=opt.batch_size, num_workers=8, pin_memory=True,
                             shuffle=False)

best_val_scores = np.zeros((opt.runs,))
best_test_scores = np.zeros((opt.runs,))
idx = list(range(119))
if opt.sim == 'cn':
    del idx[-2], idx[-1]
elif opt.sim == 'ra':
    del idx[-3], idx[-1]
elif opt.sim == 'aa':
    del idx[-2], idx[-2]

for run in range(opt.runs):
    print('********run {}*********'.format(run))
    # hit100
    np.random.seed(run)
    random.seed(run)
    torch.manual_seed(run)
    model.reset_parameters()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    best_val_score = 0.0
    best_epoch = 0
    best_test_score = 0.0
    best_model = model

    for epoch in range(opt.epochs):
        # train
        epoch_start_time = time.time()
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        model.train()
        for _, batch in enumerate(train_loader):
            edges, labels = batch
            edges = edges.cuda()
            labels = labels.cuda().squeeze()
            out = model(edges[:, idx])
            loss = lossf(out, labels)
            epoch_loss.update(loss.item(), labels.size(0))
            pred = out.argmax(dim=1)
            epoch_acc.update(torch.sum(pred == labels) / opt.batch_size, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if _ % 2000 == 0:
                print("batch {}, loss {:.4f}, acc {:.4f}".format(_, epoch_loss.avg, epoch_acc.avg))

        epoch_time = time.time() - epoch_start_time
        print("--epoch {} , loss {:.4f}, acc {:.4f}, time {:.0f}s".format(epoch, epoch_loss.avg, epoch_acc.avg,
                                                                          epoch_time))
        valid_score, test_score = test(model, pos_valid_loader, neg_valid_loader, pos_test_loader, neg_test_loader,idx)
        print('---------------------------epoch {} '.format(epoch),
              f'--Test: {100 * test_score:.2f}%, '
              f'--Val: {100 * valid_score:.2f}%')

        if best_val_score < valid_score:
            best_val_score = valid_score
            best_epoch = epoch
            best_test_score = test_score
            best_model = copy.deepcopy(model)
    best_val_scores[run] = best_val_score
    best_test_scores[run] = best_test_score
    print('********************best performance of run {} Test {} Val {}'.format(run, best_test_score, best_test_score))

log = 'Mean Val Hits: {:.4f} {:.4f}'
print(log.format(np.mean(best_val_scores), np.std(best_val_scores, ddof=1)))
log = 'Mean Test Hits: {:.4f} {:.4f}'
print(log.format(np.mean(best_test_scores), np.std(best_test_scores, ddof=1)))

torch.save(best_model, 'best_model.pkl')
