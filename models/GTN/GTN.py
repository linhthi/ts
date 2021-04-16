import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution


class GTN(nn.Module):
    """
    Using Transformer on Graph Convolutional Networks for Node Embedding
    """

    def __init__(self, in_dim=1, hidden_dim=16, out_dim=1, n_head=1,dropout=0.1):
        """

        @param in_dim: dimension of input features
        @param hidden_dim: dimension of hidden layer
        @param out_dim: dimension of output
        @param dropout: dropout layer
        """
        super(GTN, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, out_dim)
        self.fc2 = nn.Linear(out_dim*2, out_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_head, dim_feedforward=hidden_dim//4, dropout=dropout)
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, adj, nodes_u, nodes_v):
        """

        @param x: node features
        @param adj: adjacency matrix
        @return: node embedding
        """
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.gc2(h, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fc1(h).reshape(x.shape[0], 1, self.out_dim) # Node embedding
        # TODO: add transformer layer below
        h = self.transformer(h).reshape(x.shape[0], self.out_dim)
        h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
        score = self.fc2(h_uv)
        return score
