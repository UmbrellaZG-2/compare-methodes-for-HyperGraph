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
import pandas as pd

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
    return acc_train.item(), acc_val.item()

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


# 不保存结果到CSV文件


def process_dataset(dataset_path, trial_value):
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    device = torch.device("cuda" if args.gpu >= 0 else "cpu")
    H, Y, X, labels, idx_train, idx_val, idx_test, idx_pick = load_data(dataset_name)
    pairs = hypergraph_to_pairs(H)
    adj, Pv, PvT, Pe, PeT = transform(pairs)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    from utils import sparse_mx_to_torch_sparse_tensor, normalize

    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(Pv @ X.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    PvT = sparse_mx_to_torch_sparse_tensor(PvT).to(device)

    idx_train = torch.LongTensor(idx_train[trial_value].astype(np.int64)).flatten().to(device)
    idx_val = torch.LongTensor(idx_val[trial_value].astype(np.int64)).flatten().to(device)
    idx_test = torch.LongTensor(idx_test[trial_value].astype(np.int64)).flatten().to(device)

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
    train_accuracies = []
    for epoch in range(args.epochs):
        acc_train, acc_val = train(model, epoch, features, adj, PvT, labels, idx_train, idx_val, optimizer)
        train_accuracies.append(acc_train)

    total_time = time.time() - tic

    acc_test, loss_test = test(model, features, adj, PvT, labels, idx_test, dataset_name)

    return {
        'trial': trial_value,
        'dataset': dataset_name,
        'accuracy': acc_test,
        'train_accuracy': train_accuracies[-1] if train_accuracies else 0.0,
        'loss': loss_test,
        'time': total_time
    }


if args.dataset == 'all' or args.dataset is None:
    dataset_paths = load_data(dataset_str='all')
else:
    dataset_paths = [os.path.join(os.path.dirname(__file__), '..', 'data', f'{args.dataset}.mat')]

for dataset_path in dataset_paths:
    # 先加载数据获取idx_pick
    _, _, _, _, _, _, _, idx_pick = load_data(os.path.basename(dataset_path).split('.')[0])
    
    # 存储每个trial的结果
    trial_results = []
    
    for trial_idx, trial in enumerate(idx_pick):
        result = process_dataset(dataset_path, trial)
        dataset_name = result['dataset']
        
        # 存储当前trial结果
        trial_results.append({
            'trial': trial,
            'test_accuracy': result['accuracy'],
            'train_accuracy': result['train_accuracy'],
            'time': result['time']
        })
        
        print(f"Trial {trial_idx+1} (idx={trial}) completed for {dataset_name}")
    
    # 保存结果到CSV文件
    import os
    
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"LEGCN_{dataset_name}_trials.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'trial_idx', 'test_accuracy', 'train_accuracy', 'time'])
        # 写入每个trial的结果
        for trial_idx, result in enumerate(trial_results):
            writer.writerow([trial_idx + 1, result['trial'], result['test_accuracy'], result['train_accuracy'], result['time']])
    
    print(f"结果已保存到: {csv_path}")