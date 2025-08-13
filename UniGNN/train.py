import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import sys
import numpy as np
import time
import datetime
from pathlib import Path
import shutil
import random
import config
import pandas as pd


args = config.parse()


# gpu, seed
def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
fix_seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'


#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = Path( f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}' )


if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

 

# 导入数据加载函数
from data.dataloader import load_data
from prepare import initialise

# 定义fetch_data函数
def fetch_data(args):
    # 使用dataloader加载数据
    h, X, labels, idx_train_list, idx_test_list = load_data(args.dataset)
    # 转换为适当的格式
    Y = np.array([np.argmax(i) for i in labels])
    G = h
    return X, Y, G, idx_train_list, idx_test_list

# 定义accuracy函数
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


test_accs = []
best_val_accs, best_test_accs = [], []

# load data
X, Y, G, train_idx_list, test_index_list = fetch_data(args)

lst = ['trial', 'test_acc']
  # 保存结果为：项目名+数据集名
if args.model_name in ['UniGCN', 'UniGAT'] and args.add_self_loop:
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result", "UniGNN_" + args.dataset + "_self_loop.csv"))
else:
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result", "UniGNN_" + args.dataset + ".csv"))

if not os.path.exists(save_path):
    pd.DataFrame(columns=lst).to_csv(save_path, index=False, encoding='utf-8')    

for run in range(train_idx_list.shape[0]):
    run_dir = out_dir / f'{run}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # load data
    args.split = run
    train_idx = train_idx_list[run]
    test_idx = test_index_list[run]
    # Convert numpy.uint16 to int64 before creating tensor
    train_idx = torch.LongTensor(train_idx.astype(np.int64)).cuda()
    test_idx  = torch.LongTensor(test_idx.astype(np.int64)).cuda()

    # model 
    model, optimizer = initialise(X, Y, G, args)


    # baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    # baselogger.info(model)
    # baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()


    from collections import Counter
    best_test_acc, test_acc, Z = 0, 0, None    
    for epoch in range(args.epochs):
        # train
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])

        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        
        # eval
        model.eval()
        Z = model(X)
        train_acc = accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])

        # log acc
        best_test_acc = max(best_test_acc, test_acc)
        # baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')

    # 只保存最后一个epoch的结果
    if run == train_idx_list.shape[0] - 1:
        result = [run, best_test_acc]
        pd.DataFrame([result]).to_csv(save_path, index=False, mode='a+', header=False)
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


# 只打印最终平均准确率
print(f"UniGNN_{args.dataset} 平均最佳准确率: {np.mean(best_test_accs):.4f} ± {np.std(best_test_accs):.4f}")
