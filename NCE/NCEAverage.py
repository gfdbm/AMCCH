import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):
        super(NCEAverage, self).__init__()
        # 输出大小
        self.nLem = outputSize
        # 用于采样的 unigrams 张量，初始值为全1
        self.unigrams = torch.ones(self.nLem)
        # 通过 AliasMethod 进行高效的多项分布采样
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        # 模型参数
        # 负样本的数量
        self.K = K
        self.use_softmax = use_softmax
        # 注册缓冲区，包含一些参数
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        # 注册缓冲区，存储归一化后的随机权重
        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))
        # print(f'memory : {self.memory.size()}')


    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)

    # l: image组  ab: text组  y:
    def forward(self, l, ab, y, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
        # print(f'k : {K}')
        # print(f'Z_ab : {Z_ab}')
        # print(f'Z_l : {Z_l}')

        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        # 我这设置的是128
        batchSize = l.size(0)
        outputSize = self.memory.size(0) #18015
        inputSize = self.memory.size(1) # 128

        print(f'l.size : {l.size()}')
        print(f'memory.size : {self.memory.size()}')

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            print(f'idx : {idx}')
            # 将idx的第一个位置换成在数据集中的位置
            idx.select(1, 0).copy_(y.data)
            print(f'y_data : {y.data}')
            print(f'up_idx : {idx}')

        # sample
        print(f'momentum : {momentum}')
        if momentum <= 0:
            weight = (l + ab) / 2.
            print(f'weight : {weight.shape}')
            inx = torch.stack([torch.arange(batchSize)] * batchSize)

            print(f'inx : {inx}')

            inx = torch.cat([torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])], dim=1).to(weight.device).view([-1])

            print(f'inx : {inx}')
            weight = weight[inx].view([batchSize, batchSize, -1])
            print(f'up_weight : {weight.shape}')
        else:
            weight = torch.index_select(self.memory, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)
            print(f'2-weight : {weight}')

        # weight
        weight = weight.sign_()
        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        # sample
        out_l = torch.bmm(weight, l.view(batchSize, inputSize, 1))
        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l = (l + ab) / 2.
            l.div_(l.norm(dim=1, keepdim=True))
            # 使用 y 中的索引从 self.memory 中选择相应的行，将结果存储在 l_pos 中。这可能用于获取存储在 self.memory 中的先前样本的表示。
            l_pos = torch.index_select(self.memory, 0, y.view(-1))
            # 用一个称为 momentum 的常数乘以 l_pos 中的每个元素。这是一种滑动平均的操作，其中新的值占较小的比例，以保留先前的信息。
            l_pos.mul_(momentum)
            # 将 l 与 1 - momentum 的乘积加到 l_pos 上。这似乎是为了结合 l 和 l_pos，再次使用了滑动平均的思想。
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            self.memory.index_copy_(0, y, l_pos)
            print(f'new : {self.memory}')

        return out_l, out_ab
