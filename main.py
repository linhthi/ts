from PMF import PMF
from load_data import load_data
import numpy as np

__author__ = 'linhthi'

if __name__ == '__main__':
    train, validate, test, n_users, n_items = load_data('data/rating.csv')
    pmf = PMF()
    pmf.train(train, validate, n_users, n_items)
    pmf.test(test)
    # print(np.dot(U[1], V[2].T))

