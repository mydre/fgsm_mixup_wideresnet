"""toynet.py"""
import torch.nn as nn
import pdb
class ToyNet(nn.Module):
    def __init__(self, x_dim=784, y_dim=10): # 输入维度是784，输出维度是10
        super(ToyNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # mlp:multilayer perceptron 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim, 300),
            nn.ReLU(True),
            nn.Linear(300, 150),
            nn.ReLU(True),
            nn.Linear(150, self.y_dim)
            ) # 加了两个线性层，两个Relu激活函数层

    def forward(self, X):
        if X.dim() > 2: # 比如：torch.Size([100,1,28,28]) ==>  [100,728],X.size(0)相当于是batch，后面的可以类比为另一个维度
            X = X.view(X.size(0), -1)
        out = self.mlp(X)

        return out

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())
        '''
        (Pdb) self._modules
        OrderedDict([('mlp', Sequential(
          (0): Linear(in_features=784, out_features=300, bias=True)
          (1): ReLU(inplace)
          (2): Linear(in_features=300, out_features=150, bias=True)
          (3): ReLU(inplace)
          (4): Linear(in_features=150, out_features=10, bias=True)
        ))])
        '''

def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()
