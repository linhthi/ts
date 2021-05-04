from __future__ import division, print_function

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models.GTN.GTN import GTN
from models.GTN.GCN import GCN
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
    parser.add_argument('--num_neighbor', type=int, default=8)
    parser.add_argument('--num_item_neighbor', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=32)
    args = parser.parse_args()
    print(args)
    return args


def train(features, adj, train_set, model, device, optimizer):
    model.train()
    optimizer.zero_grad()
    features = nn.Embedding(features, embedding_dim=args.embedding_dim).to(device)
    adj = adj.to(device)
    score = model(features, adj, train_set[:, 0:1].reshape(train_set.shape[0], ),
                  train_set[:, 1:2].reshape(train_set.shape[0], ))
    loss_train = criterion(score, train_set[:, 2:3].type(torch.FloatTensor).to(device))
    rmse_train = torch.sqrt(loss_train)
    mae_train = mae_loss(score, train_set[:, 2:3].type(torch.FloatTensor).to(device))
    loss_train.backward()
    optimizer.step()

    return loss_train, rmse_train, mae_train


def test(features, adj_test, test_set, model, device):
    model.eval()
    features = nn.Embedding(features, embedding_dim=args.embedding_dim).to(device)
    adj_test = adj_test.to(device)
    score = model(features, adj_test, test_set[:, 0:1].reshape(len(test_set), ),
                  test_set[:, 1:2].reshape(len(test_set), ))
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
    features = torch.FloatTensor(features).to(device)
    num_train = train_set.shape[0]
    train_set = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    val_set = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=args.batch_size)
    test_set = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=args.batch_size)

    # Model and optimizer
    criterion = nn.MSELoss()
    mae_loss = nn.L1Loss()

    model_gtn = GTN(in_dim=args.embedding_dim, hidden_dim=args.hidden, out_dim=1, dropout=args.dropout)
    print(model_gtn.__repr__())
    optimizer_gtn = optim.Adam(model_gtn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_gtn.to(device)

    model_gcn = GCN(in_dim=args.embedding_dim, hidden_dim=args.hidden, out_dim=1, dropout=args.dropout)
    print(model_gcn.__repr__())
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_gcn.to(device)

    t_total = time.time()
    for epoch in range(args.epochs):
        num_iter = num_train // args.batch_size
        gtn_loss_train, gtn_rmse_train, gtn_mae_train = 0.0, 0.0, 0.0
        gcn_loss_train, gcn_rmse_train, gcn_mae_train = 0.0, 0.0, 0.0
        for i, batch in enumerate(train_set):
            start = time.time()
            batch_g, batch_set = utils.sampling_neighbor(batch, G,
                                                         n_users=n_users,
                                                         num_neighbors=args.num_neighbor,
                                                         num_items=args.num_item_neighbor)
            batch_features, batch_adj = utils.get_batches(batch_g)

            gtn_loss_train, gtn_rmse_train, gtn_mae_train = train(features=batch_features,
                                                                  adj=batch_adj,
                                                                  train_set=batch_set,
                                                                  model=model_gtn,
                                                                  device=device,
                                                                  optimizer=optimizer_gtn)
            gcn_loss_train, gcn_rmse_train, gcn_mae_train = train(features=batch_features,
                                                                  adj=batch_adj,
                                                                  train_set=batch_set,
                                                                  model=model_gcn,
                                                                  device=device,
                                                                  optimizer=optimizer_gcn)

        # Validate
        gtn_loss_val, gtn_rmse_val, gtn_mae_val = 0.0, 0.0, 0.0
        gcn_loss_val, gcn_rmse_val, gcn_mae_val = 0.0, 0.0, 0.0
        for i, val_batch in enumerate(val_set):
            val_g, val_batch_set = utils.sampling_neighbor(val_batch, G, n_users)
            val_features, val_adj = utils.get_batches(val_g)

            gtn_loss_val, gtn_rmse_val, gtn_mae_val = test(val_features, val_adj, val_batch_set, model_gtn, device)
            gcn_loss_val, gcn_rmse_val, gcn_mae_val = test(val_features, val_adj, val_batch_set, model_gcn, device)

        writer.add_scalar('GTN/loss_train', gtn_loss_train, epoch)
        writer.add_scalar('GTN/rmse_train', gtn_rmse_train, epoch)
        writer.add_scalar('GTN/mae_train', gtn_mae_train, epoch)
        writer.add_scalar('GTN/loss_val', gtn_loss_val, epoch)
        writer.add_scalar('GTN/rmse_val', gtn_rmse_val, epoch)
        writer.add_scalar('GTN/mae_val', gtn_mae_val, epoch)
        writer.add_scalar('GCN/loss_train', gcn_loss_train, epoch)
        writer.add_scalar('GCN/rmse_train', gcn_rmse_train, epoch)
        writer.add_scalar('GCN/mae_train', gcn_mae_train, epoch)
        writer.add_scalar('GCN/loss_val', gcn_loss_val, epoch)
        writer.add_scalar('GCN/rmse_val', gcn_rmse_val, epoch)
        writer.add_scalar('GCN/mae_val', gcn_mae_val, i + epoch)

        if epoch % 10 == 0:
            name = 'train/state_dict_' + str(epoch) + '_.pth'

            torch.save({
                'GCN_state_dict': model_gcn.state_dict(),
                'GTN_state_dict': model_gtn.state_dict(),
                'optimizer_gcn': optimizer_gcn.state_dict(),
                'optimizer_gtn': optimizer_gtn.state_dict(),
                "epoch": epoch,
            }, name)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    # TODO: add validation edges to predict test edges
    for i, test_batch_set in enumerate(test_set):
        test_g, test_batch_set = utils.sampling_neighbor(test_batch_set, G, n_users)
        test_features, test_adj = utils.get_batches(test_g)

        gtn_loss_test, gtn_rmse_test, gtn_mae_test = test(test_features, test_adj, test_batch_set, model_gtn, device)
        gcn_loss_test, gcn_rmse_test, gcn_mae_test = test(test_features, test_adj, test_batch_set, model_gcn, device)

        writer.add_scalar("GTN/Loss_test", gtn_loss_test, i)
        writer.add_scalar("GTN/RMSE_test", gtn_rmse_test, i)
        writer.add_scalar("GTN/MAE_test", gtn_mae_test, i)
        writer.add_scalar("GCN/Loss_test", gcn_loss_test, i)
        writer.add_scalar("GCN/RMSE_test", gcn_rmse_test, i)
        writer.add_scalar("GCN/MAE_test", gcn_mae_test, i)

    writer.close()