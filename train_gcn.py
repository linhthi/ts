from __future__ import division, print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.GTN.GCN import GCN
import utils.utils as utils
from sklearn.metrics import mean_absolute_error

# Training setting
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# TODO: Load data
training, val, test, n_users, n_items = utils.split_rating_data(args.dataset)
trust_data = utils.load_trust_network(args.dataset)
u, v = utils.get_nodes(args.dataset)
G = utils.gen_graph(training, trust_data, u, v)
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
    features.append([label_enc, label_enc])

features = torch.FloatTensor(features)
features = utils.normalize(features)

print(len(features), n_users, n_items, len(u), len(v))
targets_train, targets_val, targets_test = [], [], []
idx_train, idx_val, idx_test = [], [], []

for data in training:
    if data[0] < len(u) and data[1] < len(v):
        idx_train.append([data[0], data[1] + len(u)])
        targets_train.append([data[3]])
for data in val:
    if data[0] < len(u) and data[1] < len(v):
        idx_val.append([data[0], data[1] + len(u)])
        targets_val.append([data[3]])
for data in test:
    if data[0] < len(u) and data[1] < len(v):
        idx_test.append(([data[0], data[1] + len(u)]))
        targets_test.append([data[3]])

# targets_train = torch.LongTensor(targets_train)
# targets_val = torch.LongTensor(targets_val)
# targets_test = torch.LongTensor(targets_test)
# idx_train = torch.LongTensor(idx_train)
# idx_val  = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)
# idx1 = torch.LongTensor([0])
# idx2 = torch.LongTensor([1])


# Model and optimizer
model = GCN(in_dim=features.shape[1],
            hidden_dim=args.hidden,
            out_dim=1,
            dropout=args.dropout)
print(model.__repr__())
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
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
    output = model(features, adj)
    # print(output)
    loss = 0
    for data_train in training:
        if data_train[0] < len(u) and data_train[1] < len(v):
            y_pred = output[data_train[0]] + output[data_train[1] + len(u)]
            loss += F.mse_loss(y_pred, torch.FloatTensor(output[data_train[3]]))
            # loss.backward(retain_graph=True)
    # TODO: calculate mae, rmse metric
    mae_train = torch.sqrt(loss)
    loss.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = 0
    for data_train in val:
        if data_train[0] < len(u) and data_train[1] < len(v):
            y_pred = output[data_train[0]] + output[data_train[1] + len(u)]
            loss_val += F.mse_loss(y_pred, torch.FloatTensor(output[data_train[3]]))
    # TODO: calculate mae, rmse metric
    mae_val = torch.sqrt(loss)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss),
          'loss_val: {:.4f}'.format(loss_val),
          'mae_val: {:.4f}'.format(mae_val),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    predict = output[idx_test[:, idx1]] + output[idx_test[:, idx2]]
    print(predict.shape)
    loss_test = F.nll_loss(predict, targets_test)
    # TODO: calculate mae, rmse metric
    mae_test = mean_absolute_error(predict, targets_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          'mae_test= {:.4f}'.format(mae_test.item()),)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
