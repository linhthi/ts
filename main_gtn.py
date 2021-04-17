from __future__ import division, print_function

import time
import argparse

import torch
import torch.nn as nn
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    args = parser.parse_args()
    print(args)
    return args


def train(features, adj, train_set, model, device):
    model.train()
    optimizer.zero_grad()
    score = model(features, adj, train_set[:, 0:1].reshape(train_set.shape[0], ), train_set[:, 1:2].reshape(train_set.shape[0], ))
    loss_train = criterion(score, train_set[:, 2:3].type(torch.FloatTensor).to(device))
    rmse_train = torch.sqrt(loss_train)
    mae_train = mae_loss(score, train_set[:, 2:3].type(torch.FloatTensor).to(device))
    loss_train.backward()
    optimizer.step()

    return loss_train, rmse_train, mae_train


def test(features, adj_test, test_set, model, device):
    model.eval()
    score = model(features, adj_test, test_set[:, 0:1].reshape(len(test_set), ), test_set[:, 1:2].reshape(len(test_set), ))
    loss_test = criterion(score, test_set[:, 2:3].type(torch.FloatTensor).to(device))
    rmse_test = torch.sqrt(loss_test)
    mae_test = mae_loss(score, test_set[:, 2:3].type(torch.FloatTensor).to(device))
    return loss_test, rmse_test, mae_test


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

    adj, features, train_set, val_set, test_set, G, n_users = utils.load_data(args.dataset)
    features = torch.FloatTensor(features)
    num_train = train_set.shape[0]
    train_set = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    val_set = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=16)
    test_set = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=32)

    # Model and optimizer
    model = GTN(in_dim=1,
                hidden_dim=args.hidden,
                out_dim=1,
                dropout=args.dropout)
    print(model.__repr__())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()
    model.to(device)

    t_total = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_train, rmse_train, mae_train, loss_val, rmse_val, mae_val = 0, 0, 0, 0, 0, 0
        for i, batch in enumerate(train_set):
            start = time.time()
            batch_g, batch_set = utils.sampling_neighbor(batch, G, n_users=n_users)
            batch_features, batch_adj = utils.get_batches(batch_g)
            val_batch = [v for v in val_set]
            val_set_train = val_batch[0]
            val_g, val_set_train = utils.sampling_neighbor(val_set_train, G, n_users)
            val_features, val_adj = utils.get_batches(val_g)
            loss_train, rmse_train, mae_train = train(features=batch_features,
                                                                        adj=batch_adj,
                                                                        train_set=batch_set,
                                                                        model=model,
                                                                        device=device)
            loss_val, rmse_val, mae_val = test(val_features, val_adj, val_set_train, model, device)
            tt_time = time.time() - start
            if i % 1 == 0:
                print("Epoch{0}: {1}/{2}".format(epoch, i, num_train // args.batch_size),
                      "loss_train: {:.4f}".format(loss_train),
                      "loss_val: {:.4f}".format(loss_val),
                      "rmse_train: {:.4f}".format(rmse_train),
                      "mae_train: {:.4f}".format(mae_train),
                      "rmse_val:{:.4f}".format(rmse_val),
                      "mae_val:{:.4f}".format(mae_val),
                      "total_time: {} s".format(tt_time))
        writer.add_scalar('loss_train', loss_train, epoch)
        writer.add_scalar('rmse_train', rmse_train, epoch)
        writer.add_scalar('mae_train', mae_train, epoch)
        writer.add_scalar('rmse_val', rmse_val, epoch)
        writer.add_scalar('mae_val', mae_val, epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    # TODO: add validation edges to predict test edges
    # test_bacth_set = [t for t in test_set][0]
    for i, test_batch_set in enumerate(test_set):
        test_g, test_bacth_set = utils.sampling_neighbor(test_bacth_set, G, n_users)
        test_features, test_adj = utils.get_batches(val_g)
        loss_test, rmse_test, mae_test = test(test_features, test_adj, test_bacth_set, model, device)
        writer.add_scalar("Loss_test", loss_test, i+1)
        writer.add_scalar("RMSE_test", rmse_test, i+1)
        writer.add_scalar("MAE_test", mae_test, i+1)
    writer.close()
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test),
    #       'rmse_test= {:.4f}'.format(rmse_test),
    #       "mae_test= {:.4f}".format(mae_test))
