from models.PMF.PMF import PMF
from models.NeuMF.NeuMF import NeuMF
from utils import utils
import argparse

__author__ = 'linhthi'


# Arguments
def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='PMF')
    parser.add_argument('--dataset', nargs='?', default='ciao')
    parser.add_argument('--epochs', nargs='?', default=50, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_agrs()
    name_model = args.model
    dataset = args.dataset
    epochs = args.epochs
    train, validate, test, n_users, n_items = utils.split_rating_data(dataset)
    print(name_model)
    if name_model == 'PMF':
        pmf = PMF()
        pmf.train(train, validate, n_users, n_items, epochs)
        pmf.test(test)
        # print(np.dot(U[1], V[2].T))

    elif name_model == 'NeuMF':
        neuMF = NeuMF(number_of_users=n_users, number_of_items=n_items)
        neuMF.train(train, validate, epochs)
        neuMF.test(test)
