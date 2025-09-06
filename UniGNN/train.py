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

# 不保存结果到CSV文件

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
h, X, y, labels, train_idx_list, test_index_list, idx_pick = load_data(args.dataset)
G = H_to_G(h)
train_idx  = torch.LongTensor(train_idx_list.astype(np.int64)).cuda()
test_idx = torch.LongTensor(test_index_list.astype(np.int64)).cuda()
print("数据加载完成!")
Y = torch.LongTensor(np.where(labels)[1].astype(np.int64)).cuda()
X = torch.from_numpy(X.toarray()).float().cuda()
# 不保存结果到CSV文件

# 存储每个trial的结果
trial_results = []

for trial_idx, trial in enumerate(idx_pick):
    train_idx_run = train_idx[trial]
    test_idx_run = test_idx[trial]

    model, optimizer = initialise(X, Y, G, args)

    print(f'Trial {trial_idx+1} (idx={trial}), Total Epochs: {args.epochs}')
    print(f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tic_run = time.time()

    best_test_acc, test_acc, Z = 0, 0, None    
    train_acc_final = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx_run], Y[train_idx_run])
        loss.backward()
        optimizer.step()
        
        model.eval()
        Z = model(X)
        train_acc = accuracy(Z[train_idx_run], Y[train_idx_run])
        test_acc = accuracy(Z[test_idx_run], Y[test_idx_run])
        best_test_acc = max(best_test_acc, test_acc)
        train_acc_final = train_acc

    total_run_time = time.time() - tic_run
    print(f"Trial {trial_idx+1} (idx={trial}), best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, train accuracy: {train_acc_final:.2f}, total time: {total_run_time:.2f}s")
    
    # 存储当前trial结果
    trial_results.append({
        'trial': trial,
        'test_accuracy': test_acc,
        'train_accuracy': train_acc_final,
        'best_test_accuracy': best_test_acc,
        'time': total_run_time
    })
    
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


print(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
print(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")

# 保存结果到CSV文件
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
csv_path = os.path.join(result_dir, f"UniGNN_{args.dataset}_trials.csv")

with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'trial_idx', 'test_accuracy', 'train_accuracy', 'time'])
        # 写入每个trial的结果
        for trial_idx, result in enumerate(trial_results):
            writer.writerow([trial_idx + 1, result['trial'], result['test_accuracy'], result['train_accuracy'], result['time']])

print(f"Trial results saved to: {csv_path}")
