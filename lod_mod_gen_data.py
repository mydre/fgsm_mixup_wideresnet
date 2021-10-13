import pdb
import argparse
from datasets.datasets import return_data2
from models.model import TCN
from utils.utils import cuda
import torch
from utils.utils import str2bool
from pathlib import Path
from torch.autograd import Variable
import numpy as np
from loguru import logger

class MyGenerator(object):
    def __init__(self):
        super(MyGenerator,self).__init__()
        parser = argparse.ArgumentParser(description='get my own dataset')
        parser.add_argument('--pixel_width', type=int, default=43, help='the width of minist data picture')
        parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
        parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
        parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
        parser.add_argument('--env_name', type=str, default='main', help='experiment name')
        self.args = parser.parse_args()

        self.ckpt_dir = Path(self.args.ckpt_dir).joinpath(self.args.env_name)

        self.cuda = (self.args.cuda and torch.cuda.is_available())
        self.data_loader = return_data2(self.args)
        self.y_dim = 5
        self.channel_sizes = [25] * 8
        self.net = cuda(TCN(1, self.y_dim, self.channel_sizes, kernel_size=7, dropout=0.05), self.cuda)


    def load_checkpoint(self, filename='best_cost.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))  # 保存的tar文件是二进制的
            self.net.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def solve_train_data(self):
        idx = 0
        for images, labels in self.data_loader['train']:
            logger.info("generator train idx: " + str(idx))
            x = Variable(cuda(images, self.cuda))
            x = x.view(-1, 1, self.args.pixel_width ** 2)  # 尾部是固定的，然后有的batch的size是不完整的
            tmp, _= self.net(x)  # 网络的前半部分
            labels = labels.view(-1, 1)
            labels = labels.cpu()
            labels = labels.numpy()
            tmp = tmp.cpu()
            tmp = tmp.detach().numpy()
            data = np.concatenate((tmp, labels), axis=1)
            with open("_train.csv", "a") as f:
                np.savetxt(f, data, delimiter=',')
            idx += 1


    def solve_test_data(self):
        idx = 0
        for images, labels in self.data_loader['test']:
            logger.info("idx: " + str(idx))
            x = Variable(cuda(images, self.cuda))
            x = x.view(-1, 1, self.args.pixel_width ** 2)  # 尾部是固定的，然后有的batch的size是不完整的
            tmp, _ = self.net(x)  # 网络的前半部分
            labels = labels.view(-1, 1)
            labels = labels.cpu()
            labels = labels.numpy()
            tmp = tmp.cpu()
            tmp = tmp.detach().numpy()
            data = np.concatenate((tmp, labels), axis=1)
            with open("_test.csv", "a") as f:
                np.savetxt(f, data, delimiter=',')
            idx += 1


if __name__ == '__main__':
    myg = MyGenerator()
    myg.load_checkpoint()
    myg.solve_train_data()
    # myg.solve_test_data()