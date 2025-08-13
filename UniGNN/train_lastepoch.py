import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import scipy.io as scio
import numpy as np
import time
import datetime
import path
import shutil
import random
import config

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
out_dir = path.Path(f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

### configure logger
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)

# load data
from data import data
from prepare import *

test_accs = []
best_val_accs, best_test_accs = [], []

resultlogger.info(args)

# load data
X, Y, G, train_idx_list, test_index_list = fetch_data(args)

lst = ['trial', 'test_acc']

if args.model_name in ['UniGCN', 'UniGAT'] and args.add_self_loop:
    save_path = './results/cocitation/' + args.dataset + "_" + args.model_name + ".csv"
else:
    save_path = './results/cocitation/' + args.dataset + "_" + args.model_name + "_last_epoch.csv"

if not os.path.exists(save_path):
    pd.DataFrame(columns=lst).to_csv(save_path, index=False)

for run in range(train_idx_list.shape[0]):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # load data
    args.split = run
    train_idx = train_idx_list[run]
    test_idx = test_index_list[run]
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx = torch.LongTensor(test_idx).cuda()

    # model
    model, optimizer = initialise(X, Y, G, args)

    # baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    # baselogger.info(model)
    # baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()

    from collections import Counter

    counter = Counter(Y[train_idx].tolist())
    baselogger.info(counter)
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

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
        # best_test_acc = max(best_test_acc, test_acc)
        best_test_acc = test_acc
        # baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')

    resultlogger.info(
        f"Run {run}/{train_idx_list.shape[0]}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time() - tic_run:.2f}s")

    result = [run, best_test_acc]
    pd.DataFrame([result]).to_csv(save_path, index=False, mode='a+', header=False)
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)
scio.savemat('./results/cocitation/'+args.model_name+'_'+args.dataset+'.mat',{'acc_list':test_accs})
resultlogger.info(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
