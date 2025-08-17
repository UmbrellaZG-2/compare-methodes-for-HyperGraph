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


def save_results_to_csv(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'accuracy', 'loss', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


def process_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    print(f"\n处理数据集: {dataset_name}")

    device = torch.device("cuda" if args.gpu >= 0 else "cpu")
    H, Y, X, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    print(f"转换后标签形状: {labels.shape}, 类别数: {np.unique(labels).size}")

    pairs = hypergraph_to_pairs(H)
    adj, Pv, PvT, Pe, PeT = transform(pairs)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    from utils import sparse_mx_to_torch_sparse_tensor, normalize

    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(Pv @ X.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)
    PvT = sparse_mx_to_torch_sparse_tensor(PvT).to(device)

    idx_train = torch.LongTensor(idx_train[0].astype(np.int64)).flatten().to(device)
    idx_val = torch.LongTensor(idx_val[0].astype(np.int64)).flatten().to(device)
    idx_test = torch.LongTensor(idx_test[0].astype(np.int64)).flatten().to(device)

    adj = torch.FloatTensor(adj.toarray()).to(device)

    print(f"特征矩阵形状: {features.shape}")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"训练集大小: {len(idx_train)}, 验证集大小: {len(idx_val)}, 测试集大小: {len(idx_test)}")

    n_classes = int(labels.max().item() - labels.min().item() + 1)
    print(f"模型将使用 {n_classes} 个输出类别")

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
    print("Optimization Finished!")
    print(f"总耗时: {total_time:.4f}s")

    acc_test, loss_test = test(model, features, adj, PvT, labels, idx_test, dataset_name)

    return {
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

results = []

for dataset_path in dataset_paths:
    result = process_dataset(dataset_path)
    results.append(result)

csv_file_path = os.path.join(result_dir, 'legcn_results.csv')
save_results_to_csv(results, csv_file_path)
print(f"所有结果已保存到: {csv_file_path}")

print("\n结果摘要:")
for result in results:
    print(
        f"数据集: {result['dataset']}, 准确率: {result['accuracy']:.4f}, 损失: {result['loss']:.4f}, 时间: {result['time']:.4f}s")