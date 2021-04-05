import pandas as pd
import numpy as np
import networkx as nx
import torch
import scipy.sparse as sp

__author__ = 'linh_thi'


def split_rating_data(dataset):
    """
    Load rating_data from file csv, rating.csv includes user_id, product_id, category_id, rating, helpfulness
    and then split it to train, valid and test set.
    @param dataset:
    @return:
    """
    file_path = f'data/{dataset}/rating.csv'
    df = pd.read_csv(file_path)
    df_values = df.values

    n_users = np.max(df_values[:, 0]) + 1
    n_items = np.max(df_values[:, 1]) + 1

    train, vali, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, vali, test = train.values, vali.values, test.values

    return train, vali, test, n_users, n_items

def load_rating(dataset):
    file_path = f'data/{dataset}/rating.csv'
    return pd.read_csv(file_path).values
    
def load_trust_network(dataset):
    """
    Load trust network data from file csv
    @param dataset:
    return:
    """
    file_path = f'data/{dataset}/trustnetwork.csv'
    return pd.read_csv(file_path).values


def gen_graph(rating_data, trust_data, users, items):
    """
    Generate Graph from rating and trust network data
    @param rating_data:
    @param trust_data:
    @param users:
    @param items:
    @return: Directed graph networkx
    """
    G = nx.DiGraph()
    n_users = len(users)
    print(n_users)

    for user in users:
        G.add_node(user, id=user, label='user')

    for item in items:
        G.add_node(item + n_users, id=item + n_users, label='item')

    for data in rating_data:
        G.add_edge(data[0], data[1] + n_users, rating=data[3])

    for data in trust_data:
        G.add_edge(data[0], data[1])
    return G


def get_nodes(dataset):
    """
    Get user, item nodes
    @param dataset:
    @return:
    """
    file_path = f'data/{dataset}/rating.csv'
    df = pd.read_csv(file_path)
    users = df['user_id'].unique()
    # items = df[['item_id', 'category_id']].drop_duplicates()
    # items = items.values
    items = df['item_id'].unique()
    return users, items


def get_nodes_from_set(in_set):
    users = np.unique(in_set[:, 0:1])
    items = np.unique(in_set[:, 1:2])
    return users, items


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    @param sparse_mx:
    @return: Torch sparse
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_adjacency(G):
    return nx.to_scipy_sparse_matrix(G)


def get_adj(adj):
    """
    build symmetric adjacency matrix
    @param adj:
    @return:
    """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(dataset):
    train_set, val_set, test_set, n_users, n_items = split_rating_data(dataset)
    rating_data = load_rating(dataset)
    trust_data = load_trust_network(dataset)
    u, v = get_nodes(dataset)
    G = gen_graph(rating_data, trust_data, u, v) # Full graph

    adj = get_adj(get_adjacency(G))
    nodes = G.nodes.data()
    features = []
    for node in nodes:
        label = node[1].get('label')
        label_const = 1 if label == 'user' else 2
        features.append([label_const])
    
    idx_train = get_idx(train_set, n_users, len(nodes))
    idx_val = get_idx(val_set, n_users, len(nodes))
    idx_test = get_idx(test_set, n_users, len(nodes))

    return (adj, features, train_set, val_set, test_set, idx_train, idx_val, idx_test, n_users, len(nodes))
    
def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size

def get_idx(in_set, n_users, n_nodes):
    users = np.unique(in_set[:, 0:1])
    items = np.unique(in_set[:, 1:2])
    a = n_users*np.ones(items.shape[0])
    items = np.add(items, a)
    idx = np.unique(np.concatenate((users, items))).astype(int).tolist()
    idx_copy = []
    for i in idx:
        if i < n_nodes - 1:
            idx_copy.append(i)
    return np.array(idx_copy)

    


# Test
if __name__ == '__main__':
    training, vali, test, n_users, n_items = split_rating_data('ciao')
    # trust_data = load_trust_network('ciao')
    # u, v = get_nodes('ciao')
    # G = gen_graph(training, trust_data, u, v)
    # A = nx.to_scipy_sparse_matrix(G)
    # A = get_adj(A)
    # print(A)
    load_data('ciao')