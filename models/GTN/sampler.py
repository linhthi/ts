import pdb
import math
import torch
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import norm as sparse_norm
from torch.nn.parameter import Parameter

from utils.utils import sparse_mx_to_torch_sparse_tensor, load_data


class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'input_dim', 'layer_sizes', 'device'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, \
                'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')
        self.device = kwargs.get('device', torch.device("cpu"))

        self.num_layers = len(self.layer_sizes)

        self.adj = adj
        self.features = features

        self.train_nodes_number = self.adj.shape[0]

    def sampling(self, v_indices):
        raise NotImplementedError("sampling is not implimented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(
                sparse_mx_to_torch_sparse_tensor(adj).to(self.device))
        return new_adjs


class Sampler_GCN(Sampler):
    def __init__(self, pre_probs, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        # NOTE: uniform sampling can also has the same performance!!!!
        # try, with the change: col_norm = np.ones(features.shape[0])
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        all_support = [[]] * self.num_layers

        cur_out_nodes = v
        for layer_index in range(self.num_layers-1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes, self.layer_sizes[layer_index])
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        all_support = self._change_sparse_to_tensor(all_support)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):
        # NOTE: FastGCN described in paper samples neighboors without reference
        # to the v_indices. But in its tensorflow implementation, it has used
        # the v_indice to filter out the disconnected nodes. So the same thing
        # has been done here.
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support



if __name__ == '__main__':
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data("cora")
    batchsize = 256
    layer_sizes = [128, 128, batchsize]
    input_dim = features.shape[1]

    sampler = Sampler_GCN(None, train_features, adj_train,
                            input_dim=input_dim,
                            layer_sizes=layer_sizes, scope="None")

    batch_inds = list(range(batchsize))
    sampled_feats, sampled_adjs, var_loss = sampler.sampling(batch_inds)
    pdb.set_trace()

