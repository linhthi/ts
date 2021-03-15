import torch
import numpy as np
import torch.nn as nn
from models.GTN.GTN import GTN
import argparse
from utils import utils

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')

    args = parser.parse_args()
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    # Getting data
    rating_train, rating_valid, rating_test, n_users, n_items = utils.split_rating_data(args.dataset)
    trust_data = utils.load_trust_network(args.dataset)

    # Build graph from data
    G = utils.gen_graph(rating_train, trust_data, n_users)
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    nodes = list(G.nodes())
    node_features = torch.from_numpy(np.asarray(nodes))

    A = []
    for edge in G.edges():
        edge_tmp = torch.from_numpy(np.asarray(edge))
        value_tmp = 1
        A.append((edge_tmp, value_tmp))

    train_edge = torch.from_numpy(rating_train[:, 0:2])
    train_target = torch.from_numpy(rating_train[:, 3:4])

    valid_edge = torch.from_numpy(rating_valid[:, 0:2])
    valid_target = torch.from_numpy(rating_valid[:, 3:4])

    test_edge = torch.from_numpy(rating_test[:, 0:2])
    test_target = torch.from_numpy(rating_test[: ,3:4])

    # Regression
    num_classes = 1

    train_losses = []
    train_rmse = []
    val_losses = []
    test_losses = []
    val_rmse = []
    test_rmse = []
    final_rmse = 0

    for cnt in range(5):
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_rmse = 10000
        best_val_rmse = 10000
        best_test_rmse = 10000
        model = GTN(num_edge=num_edges,
                    num_channels=num_channels,
                    w_in=node_dim,
                    w_out=node_dim,
                    num_class=num_classes,
                    num_nodes=num_nodes,
                    num_layers=num_layers)

        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.gcn.parameters()},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {"params": model.layers.parameters(), "lr": 0.5}
                                          ], lr=0.005, weight_decay=0.001)
        loss = nn.MSELoss
        Ws = []
        for i in range(50):
            print('Epoch: ', i + 1)
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.train()
            model.zero_grad()
            loss, y_train, _ = model(A, node_features, train_edge, train_target)
            loss.backward()
            optimizer.step()
            train_rmse = RMSELoss(train_target, y_train)
            print('Train - Loss: {}, RMSE: {}'.format(loss.detach().cpu().numpy(), train_rmse))
            model.eval()

            # Valid
            with torch.no_grad():
                val_loss, y_valid, _ = model.forward(A, node_features, valid_edge, valid_target)
                val_rmse = 0
                print('Valid - Loss: {}, RMSE: {}'.format(val_loss.detach().cpu().numpy(), val_rmse))

                test_loss, y_test, W = model.forward(A, node_features, test_edge, test_target)
                test_rmse = RMSELoss(test_target, y_test)
                print('Test - Loss: {}, RMSE: {} \n'.format(test_loss.detach().cpu().numpy(), test_rmse))

                if val_rmse < best_val_rmse:
                    best_val_loss = val_loss.detach().cpu().numpy()
                    best_test_loss = test_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    best_train_rmse = train_rmse
                    best_val_rmse = val_rmse
                    best_test_rmse = test_rmse

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, RMSE: {}'.format(best_test_loss, best_train_rmse))
        print('Valid - Loss: {}, RMSE: {}'.format(best_val_loss, best_val_rmse))
        print('Test - Loss: {}, RMSE: {}'.format(best_test_loss, best_test_rmse))


