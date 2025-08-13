import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import numpy as np
import time
import datetime
import path
import shutil
import random
import config
from data.dataloader import load_data, normalize_features
from prepare import initialise, accuracy
from model import UniGNN, UniGCNII, HyperGCN


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # 解析命令行参数
    args = config.parse()
    fix_seed(args.seed)

    # 设置GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # 配置输出目录
    use_norm = 'use-norm' if args.use_norm else 'no-norm'
    add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'
    dataname = f'{args.data}_{args.dataset}'
    model_name = args.model_name
    nlayer = args.nlayer
    dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
    out_dir = path.Path(f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.makedirs_p()

    # 配置日志
    from logger import get_logger
    baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
    resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
    baselogger.info(args)

    # 加载数据 - 使用load_data函数读取已经处理好的H矩阵
    dataset_str = args.dataset  # 从命令行参数获取数据集名称
    h, X, labels, idx_train_list, idx_test_list = load_data(dataset_str)

    # 预处理特征矩阵
    X = normalize_features(X)

    # 准备数据
    X = torch.FloatTensor(np.array(X.todense())).cuda()
    Y = torch.LongTensor(np.where(labels)[1]).cuda()

    # 保存结果的路径
    lst = ['trial', 'test_acc']
    if args.model_name in ['UniGCN', 'UniGAT'] and args.add_self_loop:
        save_path = './results/' + args.dataset + "_" + args.model_name + "_self_loop.csv"
    else:
        save_path = './results/' + args.dataset + "_" + args.model_name + ".csv"

    if not os.path.exists(save_path):
        pd.DataFrame(columns=lst).to_csv(save_path, index=False)

    # 处理H矩阵以适应模型
    N, M = X.shape[0], h.shape[1]
    from torch_sparse import from_scipy
    (row, col), value = from_scipy(h)
    V, E = row, col

    # 计算度数
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum'), 'use `mean` or `sum` for first-stage aggregation'
    degV = torch.from_numpy(h.sum(1)).view(-1, 1).float().cuda()
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5).cuda()
    degV = degV.pow(-0.5).cuda()
    degV[degV.isinf()] = 1
    degE2 = torch.from_numpy(h.sum(0)).view(-1, 1).float().cuda()
    degE2 = degE2.pow(-1.).cuda()

    # 设置参数
    args.degV = degV
    args.degE = degE
    args.degE2 = degE2

    # 初始化模型
    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # 创建数据集字典
    dataset_dict = {
        'features': X.cpu().numpy(),
        'labels': labels,
        'hypergraph': None,  # 这里不需要，因为我们直接使用H矩阵
        'n': N
    }
    args.dataset_dict = dataset_dict

    test_accs = []
    best_val_accs, best_test_accs = []

    for run in range(idx_train_list.shape[0]):
        run_dir = out_dir / f'{run}'
        run_dir.makedirs_p()

        # 加载数据分割
        args.split = run
        train_idx = idx_train_list[run]
        test_idx = idx_test_list[run]
        train_idx = torch.LongTensor(train_idx).cuda()
        test_idx = torch.LongTensor(test_idx).cuda()

        # 初始化模型和优化器
        if args.model_name == 'UniGCNII':
            model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
            optimiser = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        elif args.model_name == 'HyperGCN':
            args.fast = True
            model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset_dict['n'], dataset_dict['hypergraph'], dataset_dict['features'])
            optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        else:
            model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
            optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.cuda()

        baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
        baselogger.info(model)
        baselogger.info(f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        tic_run = time.time()

        # 统计标签分布
        from collections import Counter
        counter = Counter(Y[train_idx].tolist())
        baselogger.info(counter)
        label_rate = len(train_idx) / X.shape[0]
        baselogger.info(f'label rate: {label_rate}')

        best_test_acc, test_acc, Z = 0, 0, None
        for epoch in range(args.epochs):
            # 训练
            tic_epoch = time.time()
            model.train()

            optimiser.zero_grad()
            Z = model(X)
            loss = F.nll_loss(Z[train_idx], Y[train_idx])

            loss.backward()
            optimiser.step()

            train_time = time.time() - tic_epoch

            # 评估
            model.eval()
            Z = model(X)
            train_acc = accuracy(Z[train_idx], Y[train_idx])
            test_acc = accuracy(Z[test_idx], Y[test_idx])

            # 记录最佳精度
            best_test_acc = max(best_test_acc, test_acc)
            baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')

        resultlogger.info(f"Run {run}/{idx_train_list.shape[0]}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")

        result = [run, best_test_acc]
        pd.DataFrame([result]).to_csv(save_path, index=False, mode='a+', header=False)
        test_accs.append(test_acc)
        best_test_accs.append(best_test_acc)

    resultlogger.info(f"Average final test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")