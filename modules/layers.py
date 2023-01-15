import os
import pdb
import math
import torch
import numpy as np
import torch.nn.functional as F

class FactorizedDense(torch.nn.Module):
    def __init__(self, d_i, d_o, bias=True, has_mask=True, l1=0, rank=1):
        super(FactorizedDense, self).__init__()
        self.d_i = d_i
        self.d_o = d_o
        self.has_mask = has_mask
        self.l1 = l1
        self.rank = rank
        
        self.uu = torch.nn.Parameter(torch.empty((self.d_i,self.rank), requires_grad=True, dtype=torch.float32))
        self.vv = torch.nn.Parameter(torch.empty((self.rank,self.d_o), requires_grad=True, dtype=torch.float32))  
        torch.nn.init.xavier_uniform_(self.uu)
        torch.nn.init.xavier_uniform_(self.vv)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty((self.d_o), requires_grad=True, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(self.bias.unsqueeze(0))
        else:
            self.bias = None

        if self.has_mask:
            self.mask = torch.nn.Parameter(torch.zeros((self.d_i,self.d_o), requires_grad=True, dtype=torch.float32))  
            torch.nn.init.xavier_uniform_(self.mask)

    def prune(self, mask):
        if self.training:
            return mask
        else:
            pruned = torch.abs(mask) < self.l1
            return mask.masked_fill(pruned, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.has_mask:
            weight = torch.matmul(self.uu, self.vv)
            return F.linear(input, (weight+self.prune(self.mask)).t(), self.bias)
        else:
            weight = torch.matmul(self.uu, self.vv)
            return F.linear(input, (weight).t(), self.bias)
 
class FactorizedConv(torch.nn.Module):
    def __init__(
        self,
        d_i: int,
        d_o: int,
        d_k: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        has_mask=True,
        l1=0,
        rank=1
    ):
        self.d_i = d_i
        self.d_o = d_o
        self.d_k = d_k 
        self.groups = groups 
        self.stride = tuple([stride,stride])
        self.padding = padding if isinstance(padding, str) else tuple([padding,padding])
        self.dilation = tuple([dilation,dilation])
        super(FactorizedConv, self).__init__()

        self.has_mask = has_mask
        self.l1 = l1
        
        self.rank = rank
        self.uu = torch.nn.Parameter(torch.empty((d_k*d_k,self.rank), requires_grad=True, dtype=torch.float32))
        self.vv = torch.nn.Parameter(torch.empty((self.rank,d_o*d_i), requires_grad=True, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.uu)
        torch.nn.init.xavier_uniform_(self.vv)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros((d_o,), requires_grad=True, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(self.bias.unsqueeze(0))
        else:
            self.bias = None

        if self.has_mask:
            self.mask = torch.nn.Parameter(torch.zeros((d_k*d_k, d_o*d_i), requires_grad=True, dtype=torch.float32))  
            # self.mask = torch.nn.Parameter(torch.zeros((d_o,d_i,d_k,d_k), requires_grad=True, dtype=torch.float32))  
            # torch.nn.init.xavier_uniform_(self.mask)


    def prune(self, mask):
        if self.training:
            return mask
        else:
            pruned = torch.abs(mask) < self.l1
            return mask.masked_fill(pruned, 0)

    def forward(self, input):
        if self.has_mask:
            weight = torch.matmul(self.uu, self.vv) + self.prune(self.mask)
            weight = weight.view((self.d_o,self.d_i,self.d_k,self.d_k))
        else:
            weight = torch.mul(self.uu,self.vv)
            weight = weight.view((self.d_o,self.d_i,self.d_k,self.d_k))
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



