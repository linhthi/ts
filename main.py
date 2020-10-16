from model.PMF import PMF
from model.NeuMF import NeuMF
from utils.load_data import load_data
import numpy as np
import argparse

__author__ = 'linhthi'

# Arguments
def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='')
    return parser.parse_args()

if __name__ == '__main__':
    train, validate, test, n_users, n_items = load_data('data/ciao/rating.csv')
    args = parse_agrs()
    name_model = args.model
    print(name_model)
    if name_model == 'PMF':
        pmf = PMF()
        pmf.train(train, validate, n_users, n_items)
        pmf.test(test)
        print(np.dot(U[1], V[2].T))

    elif name_model == 'NeuMF':
        neuMF = NeuMF(number_of_users=n_users, number_of_items=n_items)
        neuMF.train(train, validate)
        neuMF.test(test)

