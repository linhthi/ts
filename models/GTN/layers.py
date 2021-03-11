import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import math


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]

            edges, values = torch_sparse.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes,
                                                    self.num_nodes)
            H.append((edges, values))
        return H, W


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))
            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes,
                                                 n=self.num_nodes)
            results.append((index, value))
        return results
