from __future__ import division, print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from models.GTN.GTN import GTN
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
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(epoch, fastmode, features, train_set, n_users, n_nodes):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    count = 0
    for batch in train_set:
        idx_batch = utils.get_idx(batch.numpy(), n_users, n_nodes)
        
        batch_features = features[idx_batch]
        adj_indices = batch[:, 0:2]
        adj_values = [1 for i in range(batch.shape[0])]
        adj_batch = torch.sparse_coo_tensor(adj_indices, adj_values, (idx_batch.shape[0], idx_batch.shape[0]))
        # TODO: get adjency matrix from batch nodes (also have user-relation)
        
        score = model(batch_features, adj_batch, batch[:, 0:1], batch[:, 1:2])
        loss_train = criterion(score, batch[:, 3:4])
        count += 1
        # TODO: calculate mae metric
        rmse_train = torch.sqrt(loss_train)
        loss_train.backward()
        optimizer.step()
        print("Epoch %d/%d: loss_train: %0.4f, rmse_train: %0.4f" %(count, epoch, loss_train, rmse_train))
       
    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        adj_val_indices = val_set[:, 0:2]
        adj_val_values = [1 for i in range(val_set.shape[0])]
        adj_val = torch.sparse_coo_tensor(adj_val_indices, adj_val_values, (n_nodes, n_nodes))
        score = model(val_features, adj_val, val_set[:, 0:1], val_set[:, 1:2])

    loss_val = criterion(score, val_set[:, 3:4])
    rmse_val = torch.sqrt(loss_val)
    return (loss_train, loss_val, rmse_train, rmse_val)


def test():
    model.eval()
    adj_test_indices = val_set[:, 0:2]
    adj_test_values = [1 for i in range(test.shape[0])]
    adj_test = torch.sparse_coo_tensor(adj_test_indices, adj_test_values, (len(test_set), len(test_set)))
    score = model(tests_features, adj_test, test_set[:, 0:1], test_set[:, 1:2])

    loss_test = criterion(score, test_set[:, 3:4])
    rmse_test = torch.sqrt(loss_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          'rmse_test= {:.4f}'.format(rmse_test))


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


# Train model
if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    # Load data
    print("Loading data...")

    adj, features, train_set, val_set, test_set, idx_train, idx_val, idx_test, n_u, n_n = utils.load_data(args.dataset)
    features = torch.FloatTensor(features)
    train_features = torch.FloatTensor(features[idx_train])
    val_features = torch.FloatTensor(features[idx_val])
    tests_features = torch.FloatTensor(features[idx_test])

    print("features shape: {}".format(features.shape),
          "train_features shape: {}".format(train_features.shape),
          "val_features shape: {}".format(val_features.shape),
          "test_features shape: {}".format(tests_features.shape))

    print("train_set: {}".format(train_set.shape),
          "val_set: {}".format(val_set.shape),
          "test_set: {}".format(test_set.shape))

    print("idx_train: {}".format(idx_train.shape),
          "idx_val: {}".format(idx_val.shape),
          "idx_test: {}".format(idx_test.shape),
          "num_users: {}".format(n_u),
          "num_nodes:{}".format(n_n))

    train_set = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = torch.Tensor(val_set)
    test_set = torch.Tensor(test_set)
    print(val_set.shape)

    # Model and optimizer
    model = GTN(in_dim=1,
                hidden_dim=args.hidden,
                out_dim=1,
                dropout=args.dropout)
    print(model.__repr__())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    if device == 'cuda':
        model.cuda()
        train_features = train_features.cuda()
        val_features = val_features.cuda()
        tests_features = tests_features.cuda()

    
    t_total = time.time()
    best_rmse = 9999.0
    for epoch in range(args.epochs):
        train(epoch, args.fastmode, features, train_set, n_u, n_n)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

