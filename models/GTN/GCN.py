import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution


class GCN(nn.Module):
    """
    Simple Graph Convolutional Networks for Node Embedding with 2 Graph Convolution layers
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        """

        @param in_dim: dimension of input features
        @param hidden_dim: dimension of hidden layer
        @param out_dim: dimension of output
        @param dropout: dropout layer
        """
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        """

        @param x: node features
        @param adj: adjacency matrix
        @return: node embedding
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.log_softmax(x, dim=1)
        return x