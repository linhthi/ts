import pandas as pd
import numpy as np

__author__ = 'linhthi'


def load_data(file_path):
    """
    Load data from file csv, rating.csv includes user_id, product_id, category_id, rating, helpfulness
    :param file_path:
    :type file_path:
    :return: list of rating
    :rtype:
    """
    df = pd.read_csv(file_path)
    df_values = df.values

    n_users = np.max(df_values[:, 0])
    n_items = np.max(df_values[:, 1])

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, validate, test = train.values, validate.values, test.values

    return train, validate, test, n_users, n_items


if __name__ == '__main__':
    load_data('data/rating.csv')
