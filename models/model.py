import pdb

import torch.nn.functional as F
from torch import nn
from models.tcn import TemporalConvNet
import torch


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(num_channels[-1],output_size)
        self.linear3 = nn.Linear(num_channels[-1],output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1]) # y1[:,:,-1] == [64,25]?
        o2 = self.linear2(y1[:,:,-2])
        o3 = self.linear3(y1[:,:,-3])
        
        return F.log_softmax(o + o2/2.0 + o3/4.0, dim=1)
