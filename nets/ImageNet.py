import torch
from torch import nn
from torch.nn import functional as F


class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        """
        ImageNet类的构造函数。

        :param y_dim: 标签的维度
        :param bit: 最终二进制代码的位数
        :param norm: 一个布尔值，指示是否应用归一化（默认为True）
        :param mid_num1: 第一个隐藏层中的神经元数量（默认为1024*8）
        :param mid_num2: 第二个隐藏层中的神经元数量（默认为1024*8）
        :param hiden_layer: 隐藏层的数量（默认为3）
        """
        super(ImageNet, self).__init__()
        self.module_name = "img_model"

        # 根据隐藏层数量调整mid_num1
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        # 定义神经网络的各层
        modules = [nn.Linear(y_dim, mid_num1)]

        if hiden_layer >= 2:
            # 添加ReLU激活函数作为第一层（原地操作）。
            modules += [nn.ReLU(inplace=True)]
            # 设置第一隐藏层的输入大小。
            pre_num = mid_num1

            # 遍历隐藏层（不包括第一层和最后一层）。
            for i in range(hiden_layer - 2):
                if i == 0:
                    # 对于第一隐藏层，添加一个具有mid_num1输入大小和mid_num2输出大小的线性层，
                    # 然后是一个ReLU激活函数。
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    # 对于后续隐藏层，添加一个具有mid_num2输入和输出大小的线性层，
                    # 然后是一个ReLU激活函数。
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                # 更新下一隐藏层的输入大小。
                pre_num = mid_num2
            # 添加具有最后一个隐藏层输入大小和输出大小为'bit'的输出层。
            modules += [nn.Linear(pre_num, bit)]
        # 使用构建的模块创建一个顺序神经网络模型。
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        # 存储是否应用归一化
        self.norm = norm

    def forward(self, x):
        """
        神经网络的前向传播。

        :param x: 输入张量
        :return: 输出张量
        """
        # 应用定义的层并用tanh激活
        out = self.fc(x).tanh()
        # 如果指定了，对输出进行归一化
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
