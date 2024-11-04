import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GCN(nn.Module):
    def __init__(self, bit):
        super(GCN, self).__init__()
        self.code_len = bit

        ''' IRR_img '''


        self.gcnI1 = nn.Linear(bit, bit)
        self.BNI1 = nn.BatchNorm1d(bit)
        self.actI1 = nn.ReLU(inplace=True)


        '''IRR_txt'''


        self.gcnT1 = nn.Linear(bit, bit)
        self.BNT1 = nn.BatchNorm1d(bit)
        self.actT1 = nn.ReLU(inplace=True)


        '''CMA'''
        self.gcnJ1 = nn.Linear(bit, bit)
        self.BNJ1 = nn.BatchNorm1d(bit)
        self.actJ1 = nn.ReLU(inplace=True)



    def forward(self, XI, XT, affinity_A):

        ''' IRR_img '''
        VI = XI

        ''' IRR_txt '''
        VT = XT
        self.batch_num = XI.size(0)

###################################################################################################
        '''CMA'''
        VC = torch.cat((VI, VT), 0)
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        S_cma = torch.cat((torch.cat((affinity_A, II), 1),
                            torch.cat((II, affinity_A), 1)), 0)
####################################################################################################

        VJ1 = self.gcnJ1(VC)
        VJ1 = S_cma.mm(VJ1)
        VJ1 = self.BNJ1(VJ1)
        VJ1 = VJ1[:self.batch_num, :] + VJ1[self.batch_num:, :]
        VJ = self.actJ1(VJ1)
        return VJ
######################################################################################################1