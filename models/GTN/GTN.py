import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution


class GTN(nn.Module):
    """
    Using Transformer on Graph Convolutional Networks for Node Embedding
    """

    def __init__(self, in_dim=1, hidden_dim=16, out_dim=1, n_head=1, dropout=0.1, num_GC_layers=1):
        """

        @param in_dim: dimension of input features
        @param hidden_dim: dimension of hidden layer
        @param out_dim: dimension of output
        @param dropout: dropout layer
        @param num_GC_layers: number of Graph Convolution layers
        """
        super(GTN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.num_GC_layers = num_GC_layers
        if self.num_GC_layers == 1:
            self.gc1 = GraphConvolution(in_dim, out_dim)
        if self.num_GC_layers == 2:
            self.gc1 = GraphConvolution(in_dim, hidden_dim)
            self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.fc = nn.Linear(out_dim * 2, out_dim)
        encoder_transformer_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_head, dim_feedforward=hidden_dim,
                                                               dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer, num_layers=1)

    def forward(self, x, adj, nodes_u, nodes_v):
        """

        @param x: node features
        @param adj: adjacency matrix
        @param nodes_u: users_id
        @param nodes_v: items_id
        @return: predict scores users_id give to items_id
        """
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        if self.num_GC_layers == 2:
            h = F.relu(self.gc2(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)

        # Reshaping into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
        h = h.view(-1, 1, self.out_dim)
        h = self.transformer(h)
        h = h.view(-1, self.out_dim)
        h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
        scores = self.fc(h_uv)
        return scores
