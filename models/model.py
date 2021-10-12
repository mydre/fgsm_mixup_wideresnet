import pdb

import torch.nn.functional as F
from torch import nn
from models.tcn import TemporalConvNet
import torch


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], 100)
        self.linear2 = nn.Linear(100, 500)
        self.linear3 = nn.Linear(500, 1849)
        self.linear_nets = nn.Sequential(self.linear1, self.linear2, self.linear3)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        y1 = y1[:,:,-1] + y1[:,:,-2]/2.0 + y1[:,:,-3]/4.0  # 求这个y1和inputs之间的误差
        y2 = self.linear_nets(y1)
        return y1,y2