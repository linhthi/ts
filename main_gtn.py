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

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Training setting
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
parser.add_argument('--batch-size', default=32,
                    help='Dataset name')

args = parser.parse_args()
print(args)
np.random.seed(args.seed)

# Load data
print("Loading data...")

train_set, val_set, test_set, n_users, n_items = utils.split_rating_data(args.dataset)
trust_data = utils.load_trust_network(args.dataset)
u, v = utils.get_nodes(args.dataset)
G = utils.gen_graph(train_set, trust_data, u, v)
adj = utils.get_adjacency(G)
adj = utils.get_adj(adj)
nodes = G.nodes.data()

features = []
print(len(nodes))
for node in nodes:
    label = node[1].get('label')
    # category = 0
    label_enc = 1
    if label == 'item':
        label_enc = 2
        # category = node[1].get('category')
    features.append([label_enc])

features = torch.FloatTensor(features)
# features = torch.FloatTensor(utils.normalize(features))

print(len(features), n_users, n_items, len(u), len(v))
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set)
# test_loader = torch.utils.data.DataLoader(test_set)

# Model and optimizer
model = GTN(in_dim=1,
            hidden_dim=args.hidden,
            out_dim=1,
            dropout=args.dropout)
print(model.__repr__())
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()


# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
# targets_train = targets_train.cuda()
# targets_val = targets_val.cuda()
# targets_test = targets_test.cuda()
# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    loss_train = 0
    for data in train_set:
        if data[0] < len(u) and data[1] < len(v):
            y_pred = model(features, adj, data[0], data[1] + len(u))
            # y_pred = (output[data[0]] + output[data[1] + len(u)]).reshape(1)
            loss_train += F.mse_loss(y_pred, torch.FloatTensor([data[3]]).reshape(1))
    # TODO: calculate mae, rmse metric
    rmse_train = torch.sqrt(loss_train)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = 0
    for data in val_set:
        if data[0] < len(u) and data[1] < len(v):
            y_pred = model(features, adj, data[0], data[1] + len(u))
            # y_pred = (output[data[0]] + output[data[1] + len(u)]).reshape(1)
            loss_val += F.mse_loss(y_pred, torch.FloatTensor([data[3]]).reshape(1))
    # TODO: calculate mae, rmse metric
    rmse_val = torch.sqrt(loss_val)
    writer.add_scalar('Loss_train', loss_train, epoch)
    writer.add_scalar('rmse_train', rmse_train, epoch)
    writer.add_scalar('Loss_val', loss_val, epoch)
    writer.add_scalar('rmse_val', rmse_val, epoch)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train),
          'loss_val: {:.4f}'.format(loss_val),
          'rmse_train: {:.4f}'.format(rmse_train),
          'rmse_val: {:.4f}'.format(rmse_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = 0
    for data in test_set:
        if data[0] < len(u) and data[1] < len(v):
            y_pred = model(features, adj, data[0], data[1] + len(u))
            # y_pred = (output[data[0]] + output[data[1] + len(u)]).reshape(1)
            loss_test += F.mse_loss(y_pred, torch.FloatTensor([data[3]]).reshape(1))
            # loss.backward(retain_graph=True)
    # TODO: calculate mae, rmse metric
    rmse_test = torch.sqrt(loss_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          'rmse_test= {:.4f}'.format(rmse_test))


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


# Train model
# TODO: train with mini-batch
t_total = time.time()
best_rmse = 9999.0
for epoch in range(args.epochs):
    train(epoch)
writer.flush()
writer.close()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
