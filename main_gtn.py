from __future__ import division, print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models.GTN.GTN import GTN
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


def train(epoch, fastmode, features, train_set, val_set, idx_val, n_users, n_nodes, device, model):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    count = 0
    for batch in train_set:
        idx_batch = utils.get_idx(batch.numpy(), n_users, n_nodes)
        batch_features = features[idx_batch]
        layer_sizes = [batch_features.shape[0], batch_features.shape[0]]
        sampler = Sampler_GCN(None, features, adj, input_dim=1,
                              layer_sizes=layer_sizes, device=device)
        bf, adj_batch, _ = sampler.sampling(idx_batch)

        nodes_u, nodes_v = [], []
        idx_batch = idx_batch.tolist()
        for data in batch:
            u = int(data[0].reshape(1))
            v = int(data[1].reshape(1))
            for i in range(len(idx_batch)):
                if idx_batch[i] == u:
                    nodes_u.append(i)
                if idx_batch[i] == v + n_users:
                    nodes_v.append(i)
        score = model(batch_features.to(device), adj_batch[0].to(device), nodes_u, nodes_v)
        loss_train = criterion(score, batch[:, 3:4].type(torch.FloatTensor).to(device))
        count += 1
        # TODO: calculate mae metric
        rmse_train = torch.sqrt(loss_train)
        loss_train.backward()
        optimizer.step()
        if count % 1000 == 0:
            print("Epoch %d/%d: loss_train: %0.4f, rmse_train: %0.4f" % (count, epoch, loss_train, rmse_train))

    loss_val, rmse_val = 0, 0
    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        val_features = features[idx_val]
        bf, adj_val, _ = sampler.sampling(idx_val)

        nodes_u, nodes_v = [], []
        idx_val = idx_val.tolist()
        for data in val_set:
            u = int(data[0].reshape(1))
            v = int(data[1].reshape(1))
            for i in range(len(idx_val)):
                if idx_val[i] == u:
                    nodes_u.append(i)
                if idx_val[i] == v + n_users:
                    nodes_v.append(i)
        score = model(val_features, adj_val[0], nodes_u, nodes_v)
        loss_val = criterion(score, val_set[:, 3:4].type(torch.FloatTensor).to(device))
        rmse_val = torch.sqrt(loss_val)
    return (loss_train, loss_val, rmse_train, rmse_val)


def test(features, test_set, idx_test, n_users, model, device):
    model.eval()
    tests_features = features[idx_test]
    bf, adj_test, _ = sampler.sampling(idx_test)

    nodes_u, nodes_v = [], []
    idx_test = idx_test.tolist()
    for data in test_set:
        u = int(data[0].reshape(1))
        v = int(data[1].reshape(1))
        for i in range(len(idx_test)):
            if idx_test[i] == u:
                nodes_u.append(i)
            if idx_test[i] == v + n_users:
                nodes_v.append(i)
    score = model(tests_features.to(device), adj_test[0].to(device), nodes_u, nodes_v)
    loss_test = criterion(score, test_set[:, 3:4].type(torch.FloatTensor).to(device))
    rmse_test = torch.sqrt(loss_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          'rmse_test= {:.4f}'.format(rmse_test))


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

    adj, features, train_set, val_set, test_set, idx_train, idx_val, idx_test, n_u, n_n = utils.load_data(args.dataset)
    features = torch.FloatTensor(features)
    train_features = torch.FloatTensor(features[idx_train])
    val_features = torch.FloatTensor(features[idx_val])
    test_features = torch.FloatTensor(features[idx_test])

    print("features shape: {}".format(features.shape),
          "train_features shape: {}".format(train_features.shape),
          "val_features shape: {}".format(val_features.shape),
          "test_features shape: {}".format(test_features.shape))

    print("train_set: {}".format(train_set.shape),
          "val_set: {}".format(val_set.shape),
          "test_set: {}".format(test_set.shape))

    print("idx_train: {}".format(idx_train.shape),
          "idx_val: {}".format(idx_val.shape),
          "idx_test: {}".format(idx_test.shape),
          "num_users: {}".format(n_u),
          "num_nodes:{}".format(n_n))

    train_set = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_set = torch.Tensor(val_set).to(device)
    test_set = torch.Tensor(test_set).to(device)

    layer_sizes = [args.batch_size, args.batch_size]
    sampler = Sampler_GCN(None, features, adj, input_dim=1,
                          layer_sizes=layer_sizes, device=device)

    # Model and optimizer
    model = GTN(in_dim=1,
                hidden_dim=args.hidden,
                out_dim=1,
                dropout=args.dropout,
                sampler=sampler)
    print(model.__repr__())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    model.to(device)
    train_features.to(device)
    val_features.to(device)
    test_features.to(device)

    t_total = time.time()
    best_rmse = 9999.0
    for epoch in range(args.epochs):
        loss_train, loss_val, rmse_train, rmse_val = train(epoch, True, features, train_set,
                                                           val_set, idx_val, n_u, n_n, device, model)
        print("Epoch: {}".format(epoch),
              "loss_train: {:.4f}".format(loss_train),
              "loss_val: {:.4f}".format(loss_val),
              "rmse_train: {:.4f}".format(rmse_train),
              "rmse_val:{:.4f}".format(rmse_val))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(test_features, test_set, idx_test, n_u, model, device)
