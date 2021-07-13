"""solver.py"""
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from models.toynet import ToyNet
# from datasets.datasets import return_data
from datasets.datasets import return_data2
from utils.utils import rm_dir, cuda, where
from adversary import Attack
import pdb
from models.model import TCN
from models.arcface import ArcMarginModel


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Basic
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eps = args.eps
        self.lr = args.lr
        self.y_dim = args.y_dim
        self.target = args.target
        self.dataset = args.dataset
        # self.data_loader = return_data(args)
        self.data_loader = return_data2(args)
        self.global_epoch = 0
        self.global_iter = 0
        self.print_ = not args.silent
        self._arcface = ArcMarginModel()

        self.env_name = args.env_name
        self.tensorboard = args.tensorboard
        self.visdom = args.visdom

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization Tools
        self.visualization_init(args)

        # Histories
        self.history = dict()# self.history是一个字典
        self.history['acc'] = 0.
        self.history['epoch'] = 0
        self.history['iter'] = 0

        # Models & Optimizers
        self.model_init(args)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '':
            self.load_checkpoint(self.load_ckpt) # 加载checkpoint，恢复信息

        # Adversarial Perturbation Generator
        #criterion = cuda(torch.nn.CrossEntropyLoss(), self.cuda)
        criterion = F.cross_entropy
        self.attack = Attack(self.net, criterion=criterion)

    def visualization_init(self, args):
        # Visdom
        if self.visdom:
            from utils.visdom_utils import VisFunc
            self.port = args.visdom_port
            self.vf = VisFunc(enval=self.env_name, port=self.port)

        # TensorboardX
        if self.tensorboard:
            from tensorboardX import SummaryWriter
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists():
                self.summary_dir.mkdir(parents=True, exist_ok=True)

            self.tf = SummaryWriter(log_dir=str(self.summary_dir))
            self.tf.add_text(tag='argument', text_string=str(args), global_step=self.global_epoch)

    def model_init(self, args):
        # Network
        #self.net = cuda(ToyNet(y_dim=self.y_dim), self.cuda)
        #self.net.weight_init(_type='kaiming')
        channel_sizes = [25] * 8
        self.net = cuda(TCN(1,2,channel_sizes,kernel_size=7,dropout=0.05),self.cuda)

        # Optimizers
        #self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}],betas=(0.5, 0.999))
        self.optim = optim.Adam([{'params':self.net.parameters(), 'lr':self.lr}])
    

    def at_loss(self,x,y):
        x_adv = Variable(x.data,requires_grad=True)
        h_adv = self.net(x_adv)
        cost = F.cross_entropy(h_adv,y)
        self.optim.zero_grad()
        cost.backward()# 第一步反向传播求得梯度
        x_adv = x_adv - 0.03 * x_adv.grad#第二步根据梯度进行x值的更新
        x_adv2 = Variable(x_adv.data,requires_grad=True)
        h_adv = self.net(x_adv2)
        cost = F.cross_entropy(h_adv,y)
        return cost

    def train(self,args):
        self.set_mode('train')
        lr = args.lr
        for e in range(1,self.epoch+1):
            self.global_epoch += 1
            correct = 0.
            cost = 0.
            total = 0.
            for batch_idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1
                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                #pdb.set_trace()
                x = x.view(args.batch_size,1,784)
                '''
                经过toynet处理之后得到一个10分类的输出,logit.shape:[100,10]，所以一个batch有100个样本
                logit[0] == [0.0545  0.1646  0.0683 -0.1407  0.0031  0.0560 -0.1895 -0.0183  0.0158  0.0183】
                '''
                logit = self.net(x)
                '''
                >>> import torch
                >>> a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
                >>> print(a)
                tensor([[ 1,  5, 62, 54],
                        [ 2,  6,  2,  6],
                        [ 2, 65,  2,  6]])
                >>> torch.max(a,1)
                torch.return_types.max(
                values=tensor([62,  6, 65]),
                indices=tensor([2, 1, 1]))
                >>> torch.max(a,1)[0]
                tensor([62,  6, 65])
                >>> torch.max(a,1)[1]
                tensor([2, 1, 1])
                >>> 
                '''
                # logit.max(1)[1]其中(1)表示行的最大值，[0]表示最大的值本身,[1]表示最大的那个值在该行对应的index
                prediction = logit.max(1)[1] # prediction.shape: torch.Size([100]),此时，y == [1,2,1,1,1,3,5...],prediction也是类似的形式
                correct = torch.eq(prediction, y).float().mean().item() # 先转换为flotaTensor，然后[0]取出floatTensor中的值：0.11999999731779099
                cost = F.cross_entropy(logit, y) # cost也是一个Variable,计算出的cost是一个损失
                lds = self.at_loss(x,y)
                lds = lds * 2.0 
                cost = cost + lds
                self.optim.zero_grad()
                cost.backward()
                self.optim.step()
                if batch_idx % 200 == 0:
                    if self.print_:
                        print()
                        print(self.env_name)
                        print('[{:03d}:{:03d}]'.format(self.global_epoch, batch_idx))
                        print('acc:{:.3f} loss:{:.3f}'.format(correct, cost.item()))

                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/acc',
                                            tag_scalar_dict={'train':correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={'train':1-correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={'train':cost.data[0]},
                                            global_step=self.global_iter)
            self.test()
            # if e % 10 == 0:
            #     lr /= 10
            #     for param_group in self.optim.param_groups:
            #         param_group['lr'] = lr

        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/best/acc',
                                tag_scalar_dict={'test':self.history['acc']},
                                global_step=self.history['iter'])
        print(" [*] Training Finished!")


    def test(self):
        self.set_mode('eval')

        correct = 0.
        cost = 0.
        total = 0.

        data_loader = self.data_loader['test']
        for batch_idx, (images, labels) in enumerate(data_loader):# 相当于测试的时候也是和训练的时候一样直接通过迭代器取出来的值
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            x = x.view(-1, 1, 784)
            logit = self.net(x)
            prediction = logit.max(1)[1]

            correct += torch.eq(prediction, y).float().sum().item() # 这里不是通过mean的方式，而是通过sum的方式加在一起（即：正确的样本的个数）
            cost += F.cross_entropy(logit, y, size_average=False).item()

            total += x.size(0)

        accuracy = correct / total
        cost /= total


        if self.print_:
            print()
            print('[{:03d}]\nTEST RESULT'.format(self.global_epoch))
            print('ACC:{:.4f}'.format(accuracy))
            print('*TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy, self.global_epoch,))
            print()

            if self.tensorboard:
                self.tf.add_scalars(main_tag='performance/acc',
                                    tag_scalar_dict={'test':accuracy},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/error',
                                    tag_scalar_dict={'test':(1-accuracy)},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/cost',
                                    tag_scalar_dict={'test':cost},
                                    global_step=self.global_iter)

        if self.history['acc'] < accuracy:
            self.history['acc'] = accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint('best_acc.tar')

        self.set_mode('train')

    def generate(self, num_sample=100, target=-1, epsilon=0.03, alpha=2/255, iteration=1):
        self.set_mode('eval')

        for e in range(1):#假设有5个epoch
            self.global_epoch += 1
            for batch_idx,(x_true,y_true) in enumerate(self.data_loader['train']):# x_true: [torch.FloatTensor of size 100x1x28x28]
                self.global_iter += 1
                y_target = None
        #x_true, y_true = self.sample_data(num_sample)
        #if isinstance(target, int) and (target in range(self.y_dim)): # range(10):[0,1,2,...,9]
        #    '''
        #    (Pdb) torch.LongTensor(3).fill_(10)
        #    10
        #    10
        #    10
        #    [torch.LongTensor of size 3]
        #    '''
        #    y_target = torch.LongTensor(y_true.size()).fill_(target)
        #else:
        #    y_target = None
        ## y_target可能为None，也可能不为None
                values = self.FGSM(x_true, y_true, y_target, epsilon, alpha, iteration)
                # accuracy, cost, accuracy_adv, cost_adv = values
                correct, cost = values
                if batch_idx % 100 == 0:
                    if self.print_:
                        print()
                        print(self.env_name)
                        print('[{:03d}:{:03d}]'.format(self.global_epoch,batch_idx))
                        print('acc:{:.3f} loss:{:.3f}'.format(correct,cost.data[0]))

                    if self.tensorboard:
                        self.tf.add_scalars(main_tag='performance/acc',
                                            tag_scalar_dict={'train': correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={'train': 1 - correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={'train': cost.data[0]},
                                            global_step=self.global_iter)
            self.test()
        if self.tensorboard:
            self.tf.add_scalars(main_tag='performance/best/acc',
                                tag_scalar_dict={'test': self.history['acc']},
                                global_step=self.history['iter'])
        print(" [*] Generating Finished!")
        #save_image(x_true,
        #           self.output_dir.joinpath('legitimate(t:{},e:{},i:{}).jpg'.format(target,
        #                                                                            epsilon,
        #                                                                            iteration)),
        #           nrow=10,
        #           padding=2,
        #           pad_value=0.5)
        #save_image(x_adv,
        #           self.output_dir.joinpath('perturbed(t:{},e:{},i:{}).jpg'.format(target,
        #                                                                           epsilon,
        #                                                                           iteration)),
        #           nrow=10,
        #           padding=2,
        #           pad_value=0.5)
        #save_image(changed,
        #           self.output_dir.joinpath('changed(t:{},e:{},i:{}).jpg'.format(target,
        #                                                                         epsilon,
        #                                                                         iteration)),
        #           nrow=10,
        #           padding=3,
        #           pad_value=0.5)

        #if self.visdom:
        #    self.vf.imshow_multi(x_true.cpu(), title='legitimate', factor=1.5)
        #    self.vf.imshow_multi(x_adv.cpu(), title='perturbed(e:{},i:{})'.format(epsilon, iteration), factor=1.5)
        #    self.vf.imshow_multi(changed.cpu(), title='changed(white)'.format(epsilon), factor=1.5)

           #     print('[对抗样本，更新之后的net] accuracy : {:.2f} cost : {:.3f}'.format(accuracy, cost))
           #     print('[对抗样本，更新之前的net] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv, cost_adv))
           #     print('-----------------------------------',batch_idx)

           # self.set_mode('train')

    def sample_data(self, num_sample=100):

        total = len(self.data_loader['test'].dataset)
        '''
        >>> torch.FloatTensor(10).uniform_(1,5).long()
        tensor([1, 1, 2, 1, 3, 4, 1, 1, 4, 1])
        '''
        seed = torch.FloatTensor(num_sample).uniform_(1, total).long() # seed.shape: torch.Size([100]),等价于产生100个1到total(10000)之间的数值，如产生100个数，每个数的取值范围都是[1,10000]
        x = torch.from_numpy(self.data_loader['test'].dataset.train_set[seed]) #  self.data_loader['test'].dataset.test_data[0:2], 2x28x28
        #pdb.set_trace()
        # [100x1x28x28],
        # x = self.scale(x.float().unsqueeze(1).div(255)) #  x.float().unsqueeze(1).div(255), x.float().unsqueeze(1).div(255).mul(2).add(-1)
        x = self.scale(x.float().div(255)) #  x.float().unsqueeze(1).div(255), x.float().unsqueeze(1).div(255).mul(2).add(-1)
        y = torch.from_numpy(self.data_loader['test'].dataset.train_labels[seed]).long()# Tensor的类型设置为long
        return x, y


    def FGSM(self, x, y_true, y_target=None, eps=0.03, alpha=2/255, iteration=1): # 这个函数里面可能会调用fgsm，也可能会调用i-fgsm
        self.set_mode('eval')
        # 在转换之前x和y_true的类型都是：torch.FloatTensor
        x = Variable(cuda(x, self.cuda), requires_grad=True)


        y_true = Variable(cuda(y_true, self.cuda), requires_grad=False)
        if y_target is not None:
            targeted = True
            y_target = Variable(cuda(y_target, self.cuda), requires_grad=False)
        else:
            targeted = False


        #pdb.set_trace()
        h = self.net(x)  # h相当于是logits？
        prediction = h.max(1)[1]  # 每行最大的那个值的索引
        accuracy1 = torch.eq(prediction, y_true).float().mean()
        cost1 = F.cross_entropy(h, y_true)  # 直接logits和y_true直接计算交叉熵损失
        # print('[原始x，更新之前的net] accuracy : {:.2f} cost : {:.3f}'.format(accuracy.data[0], cost.data[0]))
        if iteration == 1: # 这里只对单个样本进行处理
            if targeted:
                x_adv, h_adv, h = self.attack.fgsm(x, y_target, True, eps)
            else:
                x_adv, h_adv, h = self.attack.fgsm(x, y_true, False, eps)
        else:
            if targeted: # 这里对多个样本进行处理，因为iterator不是1
                x_adv, h_adv, h = self.attack.i_fgsm(x, y_target, True, eps, alpha, iteration)
            else:
                x_adv, h_adv, h = self.attack.i_fgsm(x, y_true, False, eps, alpha, iteration)

        prediction = h.max(1)[1]
        accuracy = torch.eq(prediction,y_true).float().mean()
        cost = F.cross_entropy(h,y_true)
        # print('[原始x，更新之后的net] accuracy : {:.2f} cost : {:.3f}'.format(accuracy.data[0], cost.data[0]))

        # 对抗样本输入到更新之后的net
        h = self.net(x_adv)  # h相当于是logits？
        prediction = h.max(1)[1]  # 每行最大的那个值的索引
        accuracy = torch.eq(prediction, y_true).float().mean()
        cost = F.cross_entropy(h, y_true)  # 直接logits和y_true直接计算交叉熵损失

        prediction_adv = h_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv, y_true).float().mean()
        cost_adv = F.cross_entropy(h_adv, y_true)

        ## make indication of perturbed images that changed predictions of the classifier
        #if targeted:
        #    changed = torch.eq(y_target, prediction_adv)
        #else:
        #    changed = torch.eq(prediction, prediction_adv)
        #    changed = torch.eq(changed, 0)
        #changed = changed.float().view(-1, 1, 1, 1).repeat(1, 3, 28, 28)

        #changed[:, 0, :, :] = where(changed[:, 0, :, :] == 1, 252, 91)
        #changed[:, 1, :, :] = where(changed[:, 1, :, :] == 1, 39, 252)
        #changed[:, 2, :, :] = where(changed[:, 2, :, :] == 1, 25, 25)
        #changed = self.scale(changed/255)
        #changed[:, :, 3:-2, 3:-2] = x_adv.repeat(1, 3, 1, 1)[:, :, 3:-2, 3:-2]

        self.set_mode('train')

        # return (accuracy.data[0], cost.data[0], accuracy_adv.data[0], cost_adv.data[0])
        return (accuracy1.data[0], cost1)

    def save_checkpoint(self, filename='ckpt.tar'):# 保存checkpoint
        model_states = {
            'net':self.net.state_dict(),# net和Adam都有state_dict()函数
            }
        optim_states = {
            'optim':self.optim.state_dict(),
            }
        states = {
            'iter':self.global_iter,
            'epoch':self.global_epoch,
            'history':self.history,
            'args':self.args,
            'model_states':model_states,#把字典也存储进去
            'optim_states':optim_states,
            }

        file_path = self.ckpt_dir / filename
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    def scale(self, image):
        return image.mul(2).add(-1)

    def unscale(self, image):
        return image.add(1).mul(0.5)

    def summary_flush(self, silent=True):
        rm_dir(self.summary_dir, silent)

    def checkpoint_flush(self, silent=True):
        rm_dir(self.ckpt_dir, silent)
