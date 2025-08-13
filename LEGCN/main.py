import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import csv

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor
from models import GCN, SpGAT, GAT

# 设置默认参数
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
            self.dataset = None  # 默认为None，表示处理所有数据集
    return Args()

# 获取命令行参数并合并默认参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, help='Initial learning rate.')
    parser.add_argument('--fastmode', type=int, help='Validate during training pass.')
    parser.add_argument('--weight_l2', type=float, help='weight for parameter L2 regularization')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--modelType', type=int, help='GCN (0), SpGAT (1), GAT (2)')
    parser.add_argument('--dataset', type=str, help='Name of dataset, or "all" to process all datasets')

    # 获取命令行参数
    cmd_args = parser.parse_args()
    # 获取默认参数
    default_args = set_default_args()

    # 合并参数
    for attr in vars(default_args):
        if getattr(cmd_args, attr) is None:
            setattr(cmd_args, attr, getattr(default_args, attr))

    return cmd_args

args = get_args()

def train(model, epoch, features, adj, PvT, labels, idx_train, idx_val, optimizer):
    max_idx = max(idx_train).item()

    tic = time.time()
    model.train()
    optimizer.zero_grad()
    output, x = model(features, adj, PvT)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    l2 = 0
    for p in model.parameters():
        l2 = l2 + (p ** 2).sum()
    loss_train = loss_train + args.weight_l2 * l2

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, x = model(features, adj, PvT)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])

    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - tic), flush=True)


def test(model, features, adj, PvT, labels, idx_test, dataset_name):
    model.eval()
    output, x = model(features, adj, PvT)

    # 计算测试集的损失和准确率
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item(), loss_test.item()

# 保存结果到CSV文件
def save_results_to_csv(results, file_path):
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 写入CSV文件
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'accuracy', 'loss', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)

# 处理单个数据集
def process_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    print(f"\n处理数据集: {dataset_name}")
    
    # 加载数据
    adj, PvT, features, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    
    # model definition
    if args.modelType == 0:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    elif args.modelType == 1:
        adj = torch.FloatTensor(np.array(adj.todense()))
        model = SpGAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    elif args.modelType == 2:
        adj = torch.FloatTensor(np.array(adj.todense()))
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
    
    args.cuda = torch.cuda.is_available()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        Pvp = PvT.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        model.cuda()
    
    
    optimizer = optim.Adam(model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    
    # train model
    tic = time.time()
    for epoch in range(args.epochs):
        train(model, epoch, features, adj, PvT, labels, idx_train, idx_val, optimizer)
    total_time = time.time() - tic
    print("Optimization Finished!")
    print(f"Total time elapsed: {total_time:.4f}s")
    
    # test model
    acc_test, loss_test = test(model, features, adj, PvT, labels, idx_test, dataset_name)
    
    return {
        'dataset': dataset_name,
        'accuracy': acc_test,
        'loss': loss_test,
        'time': total_time
    }

# 获取要处理的数据集
if args.dataset == 'all' or args.dataset is None:
    # 获取所有mat文件路径
    dataset_paths = load_data(dataset_str='all')
else:
    # 只处理指定数据集
    dataset_paths = [os.path.join(os.path.dirname(__file__), '..', 'data', f'{args.dataset}.mat')]

# 确保结果目录存在
result_dir = os.path.join(os.path.dirname(__file__), '..', 'result')
os.makedirs(result_dir, exist_ok=True)

# 存储所有结果
results = []

# 串行处理每个数据集
for dataset_path in dataset_paths:
    result = process_dataset(dataset_path)
    results.append(result)

# 保存结果到CSV文件
csv_file_path = os.path.join(result_dir, 'legcn_results.csv')
save_results_to_csv(results, csv_file_path)
print(f"所有结果已保存到: {csv_file_path}")

# 打印所有结果摘要
print("\n结果摘要:")
for result in results:
    print(f"数据集: {result['dataset']}, 准确率: {result['accuracy']:.4f}, 损失: {result['loss']:.4f}, 时间: {result['time']:.4f}s")
