import numpy as np
import math

__author__ = 'linhthi'


class PMF(object):
    def __init__(self, learning_rate=0.1, k=10, lam_u=0.01, lam_i=0.01):
        """
        Probabilistic Matrix Factorization model
        :param learning_rate:
        :param k: latent dimension
        :param lam_u:
        :param lam_i:
        U: User feature matrix [mxk]
        V: Item feature matrix [nxk]
        """

        self.lam_i = lam_i
        self.lam_u = lam_u
        self.learning_rate = learning_rate
        self.K = k
        self.U = None
        self.V = None

    def train(self, train, test, num_user, num_item):
        """
        Fit model with training set and evaluate RMSE/MAE on test set
        :param train: training_set
        :param test: test_set
        :param num_user: number of user
        :param num_item: number of item
        :return:
        """
        self.U = np.random.normal(0, 0.1, (num_user + 1, self.K))
        self.V = np.random.normal(0, 0.1, (num_item + 1, self.K))
        pre_rmse = 100.0
        endure_count = 0
        patience = 5
        epoch = 0
        R_max = 5
        while endure_count < patience:
            loss = 0.0
            for data in train:
                user = data[0]
                item = data[1]
                rating = data[3]

                predict_rating = np.dot(self.U[user], self.V[item].T)
                error = f(rating, R_max) - sigmoid(predict_rating)
                loss += error ** 2

                # Update U, V by gradient descent
                self.U[user] += self.learning_rate * (error * self.V[item] - self.lam_u * self.U[user])
                self.V[item] += self.learning_rate * (error * self.U[user] - self.lam_i * self.V[item])

                loss += self.lam_u * np.square(self.U[user]).sum() \
                        + self.lam_i * np.square(self.V[item]).sum()
            loss = 0.5 * loss
            rmse, mae = self.eval_metric(test)
            epoch += 1
            print('Epoch:%d loss:%.3f rmse:%.5f mae:%.5f' % (epoch, loss, rmse, mae))
            if rmse < pre_rmse:
                pre_rmse = rmse
                endure_count = 0
            else:
                endure_count += 1

    def eval_metric(self, test):
        """
        Get MAE, RMSE
        :param test:
        :return:
        """
        test_count = len(test)
        tmp_rmse = 0.0
        tmp_mae = 0.0
        for te in test:
            user = te[0]
            item = te[1]
            real_rating = te[3]
            predict_rating = np.dot(self.U[user], self.V[item].T)
            tmp_rmse += np.square(f(real_rating, 5) - sigmoid(predict_rating))
            tmp_mae += np.abs(f(real_rating, 5) - sigmoid(predict_rating))
        rmse = np.sqrt(tmp_rmse / test_count)
        mae = (1 / test_count) * tmp_mae
        return rmse, mae


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def f(x, r_max):
    """
    Map rating [1, r-max] to [0, 1]
    """
    return (x - 1) / (r_max - 1)
