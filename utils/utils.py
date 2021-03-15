import pandas as pd
import numpy as np
import networkx as nx

__author__ = 'linhthi'


def split_rating_data(dataset):
    """
    Load rating_data from file csv, rating.csv includes user_id, product_id, category_id, rating, helpfulness
    and then split it to train, valid and test set.
    :param file_path:
    :type file_path:
    :return: train, vali, test set, number of users and number of item
    :rtype:
    """
    file_path = f'data/{dataset}/rating.csv'
    df = pd.read_csv(file_path)
    df_values = df.values

    n_users = np.max(df_values[:, 0]) + 1
    n_items = np.max(df_values[:, 1]) + 1

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, validate, test = train.values, validate.values, test.values

    return train, validate, test, n_users, n_items


def load_trust_network(dataset):
    """
    Load trust network data from file csv
    @param file_path: example: 'data/ciao/trustnetwork.csv'
    return:
    """
    file_path = f'data/{dataset}/trustnetwork.csv'
    return pd.read_csv(file_path).values


def gen_graph(rating_data, trust_data, n_users):
    """
    Generate Graph from rating and trust network data
    @param rating_data:
    @param trust_data:
    """
    G = nx.Graph()

    for data in rating_data:
        G.add_node(data[0], id=data[0], label='user')
        G.add_node(data[1] + n_users, id=data[1] + n_users, label='item')
        G.add_edge(data[0], data[1] + n_users, rating=data[3])

    for data in trust_data:
        G.add_edge(data[0], data[1])
    return G


