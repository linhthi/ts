from __future__ import division, print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models.GTN.GCN import GCN
from models.GTN.sampler import Sampler_GCN
import utils.utils as utils
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', default='ciao',
                        help='Dataset name')
    parser.add_argument('--batch_size', default=32,
                        help='Dataset name')
    args = parser.parse_args()
    print(args)
    return args


def train(features, adj, train_set, val_set, fastmode, model, device):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    score = model(features, adj, train_set[:, 0:1].reshape(train_set.shape[0], ), train_set[:, 1:2].reshape(train_set.shape[0], ))
    loss_train = criterion(score, train_set[:, 3:4].type(torch.FloatTensor).to(device))
    rmse_train = torch.sqrt(loss_train)
    mae_train = mae_loss(score, train_set[:, 3:4].type(torch.FloatTensor).to(device))
    loss_train.backward()
    optimizer.step()

    loss_val, rmse_val, mae_val = 0, 0, 0
    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        score = model(features, adj, val_set[:, 0:1].reshape(val_set.shape[0], ), val_set[:, 1:2].reshape(val_set.shape[0], ))
        loss_val = criterion(score, val_set[:, 3:4].type(torch.FloatTensor).to(device))
        rmse_val = torch.sqrt(loss_val)
        mae_val = mae_loss(score, val_set[:, 3:4].type(torch.FloatTensor).to(device))
    total_time = time.time() - t
    return loss_train, loss_val, rmse_train, mae_train, rmse_val, mae_val, total_time


def test(features, adj_test, test_set, model, device):
    model.eval()
    score = model(features, adj_test, test_set[:, 0:1].reshape(len(test_set), ), test_set[:, 1:2].reshape(len(test_set), ))
    loss_test = criterion(score, test_set[:, 3:4].type(torch.FloatTensor).to(device))
    rmse_test = torch.sqrt(loss_test)
    mae_test = mae_loss(score, test_set[:, 3:4].type(torch.FloatTensor).to(device))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          'rmse_test= {:.4f}'.format(rmse_test),
          "mae_test= {:.4f}".format(mae_test))


# Train model
if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    # Load data
    print("Loading data...")

    adj, features, train_set, val_set, test_set, G = utils.load_data(args.dataset)
    features = torch.FloatTensor(features)
    train_set = torch.LongTensor(train_set).to(device)
    val_set = torch.LongTensor(val_set).to(device)
    test_set = torch.LongTensor(test_set).to(device)

    # Model and optimizer
    model = GCN(in_dim=1,
                hidden_dim=args.hidden,
                out_dim=1,
                dropout=args.dropout)
    print(model.__repr__())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    model.to(device)

    t_total = time.time()
    for epoch in range(args.epochs):
        loss_train, loss_val, rmse_train, mae_train, rmse_val, mae_val, tt_time = train(features=features,
                                                                    adj=adj,
                                                                    train_set=train_set,
                                                                    val_set=val_set,
                                                                    fastmode=args.fastmode,
                                                                    # fastmode=True,
                                                                    model=model,
                                                                    device=device)
        print("Epoch: {}".format(epoch),
              "loss_train: {:.4f}".format(loss_train),
              "loss_val: {:.4f}".format(loss_val),
              "rmse_train: {:.4f}".format(rmse_train),
              "mae_train: {:.4f}".format(mae_train),
              "rmse_val:{:.4f}".format(rmse_val),
              "mae_val:{:.4f}".format(mae_val),
              "total_time: {} s".format(tt_time))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    # TODO: add validation edges to predict test edges
    test(
        features=features,
        adj_test=adj,
        test_set=test_set,
        model=model,
        device=device)
