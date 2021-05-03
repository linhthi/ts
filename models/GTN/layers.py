import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Graph Convolution layer
    """

    def __init__(self, in_dim: int, out_dim: int, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return '{}(in_dim={}, out_dim={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)
