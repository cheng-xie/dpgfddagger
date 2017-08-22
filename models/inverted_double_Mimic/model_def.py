import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(torch.nn.Module):
    """Defines custom model
    Inherits from torch.nn.Module
    """
    def __init__(self, dim_input, dim_output):

        super(Actor, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output

        SIZE_H1 = 50
        SIZE_H2 = 20
        # SIZE_H3 = 60

        '''Initialize nnet layers'''
        self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
        self._l2 = torch.nn.Linear(SIZE_H1, SIZE_H2)
        self._l3 = torch.nn.Linear(SIZE_H2, self._dim_output)
        # self._l4 = torch.nn.Linear( SIZE_H3, self._dim_output)

    def forward(self,s_t):
        x = s_t # hVariable(torch.FloatTensor(s_t.astype(np.float32)))
        #print(s_t)
        self._l1_out = F.relu(self._l1(x))
        self._l2_out = F.relu(self._l2(self._l1_out))
        self._out = (self._l3(self._l2_out))

        #print('_out',self._out)
        return self._out

"""
class Critic(nn.Module):
    def __init__(self,dim_input, dim_output):
        super(Critic, self).__init__()
        self._dim_input = dim_input
        self._dim_output = dim_output

        H_LAYER1 = 50
        H_LAYER2 = 20
        # H_LAYER4 = 10

        self.linear1 = nn.Linear(self._dim_input, H_LAYER1)
        self.linear2 = nn.Linear(H_LAYER1, H_LAYER2)
        self.linear3 = nn.Linear(H_LAYER2, self._dim_output)
        # self.linear5 = nn.Linear(H_LAYER4, self._dim_output)

    def forward(self,s,a):
        '''
        s = Variable(torch.FloatTensor(np.array(s,dtype=np.float32)))
        if(type(a)!=type(s)):
            a = Variable(torch.FloatTensor(np.array(a,dtype=np.float32)))
        '''
        x = torch.cat([s,a],1)

        a1 = F.relu(self.linear1(x))
        a2 = F.relu(self.linear2(a1))
        y = self.linear3(a2)
        return y
"""
