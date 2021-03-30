import pandas as pd
import numpy as np
import networkx as nx
import torch

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
        G.add_node(item[0] + n_users, id=item[0] + n_users, label='item', category=item[1])

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
    items = df[['item_id', 'category_id']].drop_duplicates()
    items = items.values
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

# Test
if __name__ == '__main__':
    training, vali, test, n_users, n_items = split_rating_data('ciao')
    trust_data = load_trust_network('ciao')
    u, v = get_nodes('ciao')
    G = gen_graph(training, trust_data, u, v)
    A = nx.to_scipy_sparse_matrix(G)
    print(A)
