import time
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import csv

from LE import hypergraph_to_pairs, transform
from utils import load_data, accuracy
from models import GCN, SpGAT, GAT


def set_default_args():
    class Args:
        def __init__(self):
            self.seed = 200
            self.epochs = 50
            self.lr = 0.02
            self.fastmode = 0
            self.weight_l2 = 1.5e-3
            self.weight_decay = 5e-3
            self.hidden = 256
            self.dropout = 0.5
            self.modelType = 0
            self.dataset = None

    return Args()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate.')
    parser.add_argument('--fastmode', type=int, default=0, help='Validate during training pass.')
    parser.add_argument('--weight_l2', type=float, default=1.5e-3, help='weight for parameter L2 regularization')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--modelType', type=int, default=0, help='GCN (0), SpGAT (1), GAT (2)')
    parser.add_argument('--dataset', type=str, default=None, help='Name of dataset, or "all" to process all datasets')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id to use, -1 for CPU')

    cmd_args = parser.parse_args()
    return cmd_args


args = get_args()


def train(model, epoch, features, adj, PvT, labels, idx_train, idx_val, optimizer):
    tic = time.time()
    model.train()
    optimizer.zero_grad()
    output,x = model(features, adj, PvT)

    idx_train = idx_train.to(torch.long)
    idx_val = idx_val.to(torch.long)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    l2 = 0
    for p in model.parameters():
        l2 = l2 + (p ** 2).sum()
    loss_train = loss_train + args.weight_l2 * l2

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output,x = model(features, adj, PvT)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])

    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - tic), flush=True)


def test(model, features, adj, PvT, labels, idx_test, dataset_name):
    model.eval()
    output, x = model(features, adj, PvT)

    idx_test = idx_test.to(torch.long)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item(), loss_test.item()


def save_results_to_csv(result, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 检查文件是否存在
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['trial', 'accuracy', 'total_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        # 构建要写入的数据，确保字段匹配
        row_data = {
            'trial': result['trial'] + 1,  # trial从1开始计数
            'accuracy': result['accuracy'],
            'total_time': result['time']  # 将time改为total_time
        }
        writer.writerow(row_data)


def process_dataset(dataset_path, x ):
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    device = torch.device("cuda" if args.gpu >= 0 else "cpu")
    H, Y, X, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    pairs = hypergraph_to_pairs(H)
    adj, Pv, PvT, Pe, PeT = transform(pairs)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    from utils import sparse_mx_to_torch_sparse_tensor, normalize

    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(Pv @ X.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    PvT = sparse_mx_to_torch_sparse_tensor(PvT).to(device)

    idx_train = torch.LongTensor(idx_train[x].astype(np.int64)).flatten().to(device)
    idx_val = torch.LongTensor(idx_val[x].astype(np.int64)).flatten().to(device)
    idx_test = torch.LongTensor(idx_test[x].astype(np.int64)).flatten().to(device)

    adj = torch.FloatTensor(adj.toarray()).to(device)
    n_classes = int(labels.max().item() - labels.min().item() + 1)

    if args.modelType == 0:
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=n_classes,
                    dropout=args.dropout)
    elif args.modelType == 1:
        model = SpGAT(nfeat=features.shape[1],
                      nhid=args.hidden,
                      nclass=n_classes,
                      dropout=args.dropout)
    elif args.modelType == 2:
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=n_classes,
                    dropout=args.dropout)

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    tic = time.time()
    for epoch in range(args.epochs):
        train(model, epoch, features, adj, PvT, labels, idx_train, idx_val, optimizer)

    total_time = time.time() - tic

    acc_test, loss_test = test(model, features, adj, PvT, labels, idx_test, dataset_name)

    return {
        'trial': x,
        'dataset': dataset_name,
        'accuracy': acc_test,
        'loss': loss_test,
        'time': total_time
    }


if args.dataset == 'all' or args.dataset is None:
    dataset_paths = load_data(dataset_str='all')
else:
    dataset_paths = [os.path.join(os.path.dirname(__file__), '..', 'data', f'{args.dataset}.mat')]

result_dir = os.path.join(os.path.dirname(__file__), '..', 'result')
os.makedirs(result_dir, exist_ok=True)

for dataset_path in dataset_paths:
    for x in range(1000):
        result = process_dataset(dataset_path,x)
        dataset_name = result['dataset']
        csv_file_path = os.path.join(result_dir, f'LEGCN_{dataset_name}.csv')
        save_results_to_csv(result, csv_file_path)