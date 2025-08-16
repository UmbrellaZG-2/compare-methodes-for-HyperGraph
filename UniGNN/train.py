import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import scipy.sparse as sp

import numpy as np
import time
import datetime
import shutil
import random
import config
from pathlib import Path

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
Y = torch.LongTensor(np.where(labels)[1]).cuda()
X = torch.from_numpy(X.toarray()).float().cuda()
lst = ['trial', 'test_acc']
result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result"))
os.makedirs(result_dir, exist_ok=True)

if args.model_name in ['UniGCN', 'UniGAT'] and args.add_self_loop:
    save_path = os.path.join(result_dir, f"{args.dataset}_{args.model_name}_self_loop.csv")
else:
    save_path = os.path.join(result_dir, f"{args.dataset}_{args.model_name}.csv")

if not os.path.exists(save_path):
    pd.DataFrame(columns=lst).to_csv(save_path, index=False)

for run in range(1):
    out_dir = Path(args.out_dir)
    run_dir = out_dir / f'{run}'
    run_dir.mkdir(parents=True, exist_ok=True)

    args.split = run
    train_idx = train_idx_list[run]
    test_idx = test_index_list[run]


    model, optimizer = initialise(X, Y, G, args)


    print(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    print(model)
    print(f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tic_run = time.time()
    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    print(counter)
    label_rate = len(train_idx) / X.shape[0]
    print(f'label rate: {label_rate}')

    best_test_acc, test_acc, Z = 0, 0, None    
    for epoch in range(args.epochs):
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])

        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        
        model.eval()
        Z = model(X)
        train_acc = accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])

        best_test_acc = max(best_test_acc, test_acc)

    print(f"Run {run}/{train_idx_list.shape[0]}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")

    result = [run, best_test_acc]
    pd.DataFrame([result]).to_csv(save_path, index=False, mode='a+', header=False)
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


print(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
print(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
