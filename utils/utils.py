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
    @param dataset: dataset's name. Ex: 'ciao'.
    @return: train, vali, test, n_user, n_items
    """
    file_path = f'{dataset}/rating.csv'
    df = pd.read_csv(file_path)
    df_values = df.values

    n_users = np.max(df_values[:, 0]) + 1
    n_items = np.max(df_values[:, 1]) + 1

    train, vali, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, vali, test = train.values, vali.values, test.values

    return train, vali, test, n_users, n_items


def load_rating(dataset):
    """
    Load data from rating csv file
    @param dataset: dataset's name. Ex: 'ciao'.
    @return:
    """
    file_path = f'{dataset}/rating.csv'
    return pd.read_csv(file_path).values


def load_trust_network(dataset):
    """
    Load trust network data from file csv
    @param dataset: dataset's name. Ex: 'ciao'
    return:
    """
    file_path = f'{dataset}/trustnetwork.csv'
    return pd.read_csv(file_path).values


def gen_graph(rating_data, trust_data, n_users, n_items):
    """
    Generate Graph from rating and trust network data
    @param rating_data: contain user_id, item_id, category_id, rating, helpfulness
    @param trust_data: contain user_id, user_trust_id
    @param n_users: number of users in graph
    @param n_items: number of items in graph
    @return: Directed graph networkx
    """
    G = nx.DiGraph()
    print(n_users, n_items)

    for user in range(1, n_users):
        G.add_node(user, id=user, label='user')

    for item in range(n_users, n_items + n_users):
        G.add_node(item + n_users, id=item, label='item')

    for data in rating_data:
        G.add_edge(data[0], data[1] + n_users - 1, rating=data[3])

    for data in trust_data:
        G.add_edge(data[0], data[1])
    return G


def get_nodes(dataset):
    """
    Get user, item nodes
    @param dataset: dataset's name
    @return: users, items type np.array
    """
    file_path = f'{dataset}/rating.csv'
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


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_adjacency(G):
    """
    Get sparse adjacency matrix from graph G
    @param G: networkx Graph
    @return: sparse adjacency matrix
    """
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
    G = gen_graph(rating_data, trust_data, n_users, n_items)  # Full graph

    adj = get_adj(get_adjacency(G))
    # adj = nx.adj_matrix(G)
    # adj = normalize(adj)
    nodes = G.nodes.data()
    features = []
    for node in nodes:
        label = node[1].get('label')
        label_const = 1 if label == 'user' else 2
        features.append([label_const])

    return adj, features, train_set, val_set, test_set, G, n_users


def get_batches(graph):
    """
    Get features of nodes from subgraph
    @param graph: subgraph: type networkx Graph
    @return: Tensor
    """
    adj = get_adj(get_adjacency(graph))
    nodes = graph.nodes.data()
    features = []
    for node in nodes:
        # id = node[1].get('id')
        label = node[1].get('label')
        label_const = 1 if label == 'user' else 2
        features.append([label_const])
    return torch.Tensor(features), adj


def get_idx(in_set, n_users):
    """
    Get index list from set node
    @param in_set:
    @param n_users: number of users in full-graph
    @return: index list: type nparray
    """
    users = np.unique(in_set[:, 0:1])
    items = np.unique(in_set[:, 1:2])
    a = n_users * np.ones(items.shape[0])
    items = np.add(items, a)
    idx = np.unique(np.concatenate((users, items))).astype(int).tolist()
    return np.array(idx)


def sampling_neighbor(batch, full_graph, n_users, num_neighbors=8, num_items=4):
    """
    Sampling neighbor nodes with users and items from full graph and reindex nodes
        @param batch: contain item, user, rating for training
        @param full_graph: G networkx graph
        @param n_users: total user in graph G
        @param num_neighbors: number of neighbor's node
        @param num_items: limitation of item that neighbor have bought
    """
    users_id = np.unique(batch[:, 0:1]).tolist()
    items_id = np.unique(batch[:, 1:2]).tolist()
    users_map = dict()
    items_map = dict()
    G = nx.DiGraph()

    # Add user nodes and items which friend bought
    num_node_friends, num_node_items_friend_bought = 0, 0

    for user in range(1, len(users_id)+1):
        G.add_node(user, id=user, label='user')
        users_map.update({users_id[user-1] : user})
        cnt_friend = 0
        friends = [n for n in full_graph.neighbors(user) if n <= n_users]

        while cnt_friend < len(friends) and cnt_friend < num_neighbors:
            friend_node = user + num_node_friends + len(users_id) + len(items_id)
            G.add_node(friend_node, label='user')
            friend = friends[cnt_friend]
            cnt_friend += 1
            num_node_friends += 1
            items_friend_bought = [n for n in full_graph.neighbors(friend) if n > n_users]
            cnt_item = 0
            while cnt_item < len(items_friend_bought) and cnt_item < num_items:
                item_node = user + 4 * len(users_id) + len(items_id) + num_node_items_friend_bought
                G.add_node(item_node, label='item')
                item = items_friend_bought[cnt_item]
                G.add_edge(friend_node, item_node, rating=full_graph[friend][item])
                cnt_item += 1
                num_node_items_friend_bought += 1

    cnt = 0
    for item in range(len(users_id), len(users_id) + len(items_id)):
        G.add_node(item, id=item, label='item')
        items_map.update({items_id[cnt] + len(users_id) : item})
        cnt += 1

    train_set = []
    for data in batch:
        u = data[0]
        v = data[1] + len(users_id)
        G.add_edge(u, v, rating=data[3])
        train_set.append([users_map.get(int(u)), items_map.get(int(v)), data[3]])

    return G, torch.LongTensor(train_set)