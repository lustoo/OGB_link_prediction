import argparse
import torch
from tqdm import tqdm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_hits(evaluator, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results





def get_opt():
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--sim', type=str, default='all', choices=['all', 'ra', 'aa', 'cn'])
    opt = parser.parse_args()
    return opt
