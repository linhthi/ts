from PMF import PMF
from load_data import load_data

__author__ = 'linhthi'

if __name__ == '__main__':
    train, validate, test, n_users, n_items = load_data('data/rating.csv')
    pmf = PMF()
    pmf.train(train, test, n_users, n_items)
