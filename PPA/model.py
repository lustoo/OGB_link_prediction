import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, class_num=2):
        hidden1 = 256
        hidden2 = 512
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden2, class_num)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

    def reset_parameters(self):
        self.linear[0].reset_parameters()
        self.linear[2].reset_parameters()
        self.linear[5].reset_parameters()

