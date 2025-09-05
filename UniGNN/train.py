import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import scipy.sparse as sp
import csv

import numpy as np
import time
import datetime
import shutil
import random
import config
from pathlib import Path

# 结果保存函数
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

args = config.parse()

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
fix_seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'


from prepare import H_to_G, load_data, normalize_features, initialise, accuracy

test_accs = []
best_val_accs, best_test_accs = [], []

print("开始加载数据...")
h, X, y, labels, train_idx_list, test_index_list = load_data(args.dataset)
G = H_to_G(h)
train_idx  = torch.LongTensor(train_idx_list.astype(np.int64)).cuda()
test_idx = torch.LongTensor(test_index_list.astype(np.int64)).cuda()
print("数据加载完成!")
Y = torch.LongTensor(np.where(labels)[1].astype(np.int64)).cuda()
X = torch.from_numpy(X.toarray()).float().cuda()
result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))
os.makedirs(result_dir, exist_ok=True)

# 统一使用{方法名+数据集名}的格式命名文件
method_name = 'UniGNN'
save_path = os.path.join(result_dir, f"{method_name}_{args.dataset}.csv")

for run in range(1):
    train_idx_run = train_idx[run]
    test_idx_run = test_idx[run]

    model, optimizer = initialise(X, Y, G, args)

    print(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    print(f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tic_run = time.time()

    # 创建详细记录文件
    detailed_csv_path = os.path.join(result_dir, "detailed", f"UniGNN_{args.dataset}_trial{run+1}_epochs.csv")
    os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)
    
    epoch_data = []
    
    best_test_acc, test_acc, Z = 0, 0, None    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 前向传播
        forward_start = time.time()
        model.train()
        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx_run], Y[train_idx_run])
        forward_time = time.time() - forward_start
        
        # 反向传播
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        
        # 评估
        model.eval()
        Z = model(X)
        train_acc = accuracy(Z[train_idx_run], Y[train_idx_run])
        test_acc = accuracy(Z[test_idx_run], Y[test_idx_run])
        best_test_acc = max(best_test_acc, test_acc)
        
        # 记录epoch总时间
        total_epoch_time = time.time() - epoch_start_time
        
        # 保存epoch详细数据
        epoch_data.append({
            'epoch': epoch + 1,
            'forward_pass_time': forward_time,
            'loss_calc_time': 0.0,  # 在UniGNN中损失计算时间包含在前向传播中
            'backward_pass_time': backward_time,
            'total_epoch_time': total_epoch_time,
            'train_accuracy': train_acc * 100,
            'test_accuracy': test_acc * 100
        })

    total_run_time = time.time() - tic_run
    
    # 保存详细记录
    if epoch_data:
        pd.DataFrame(epoch_data).to_csv(detailed_csv_path, index=False)
    print(f"Run {run}/{train_idx_list.shape[0]}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {total_run_time:.2f}s")

    # 构建结果字典
    result = {
        'trial': run,
        'accuracy': best_test_acc,
        'loss': 0,  # 这里可以根据实际情况计算loss
        'time': total_run_time,
    }
    
    # 使用save_results_to_csv函数保存结果
    save_results_to_csv(result, save_path)
    
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


print(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
print(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
