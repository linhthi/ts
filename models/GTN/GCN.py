import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution


class GCN(nn.Module):
    """
    Simple Graph Convolutional Networks for Node Embedding with 2 Graph Convolution layers
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_GC_layers=2):
        """

        @param in_dim: dimension of input features
        @param hidden_dim: dimension of hidden layer
        @param out_dim: dimension of output
        @param num_GC_layers: number of Graph Convolution layers with value: 1 or 2, default 2
        @param dropout: dropout layer
        """
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_GC_layers = num_GC_layers
        self.dropout = dropout
        if self.num_GC_layers == 1:
            self.gc1 = GraphConvolution(in_dim, out_dim)
        if self.num_GC_layers == 2:
            self.gc1 = GraphConvolution(in_dim, hidden_dim)
            self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.fc = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x, adj, nodes_u, nodes_v):
        """

        @param x: node features
        @param adj: adjacency matrix
        @return: node embedding
        """
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        if self.num_GC_layers == 2:
            h = F.relu(self.gc2(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
        scores = self.fc(h_uv)
        return scores


