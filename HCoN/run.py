import numpy as np
from utils import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric
from model import HCoN
import torch
from torch import optim
import torch.nn.functional as F
import os
import argparse
import numpy as np


def training(data, args, s=2021, trial_id=0, dataset_name=""):
    import time
    import pandas as pd
    import os

    seed_everything(seed=s)

    H_trainX = torch.from_numpy(data.H_trainX.toarray()).float().to(device)
    X = torch.from_numpy(data.X.toarray()).float().to(device)
    Y = torch.from_numpy(data.Y).float().to(device)

    hx1 = torch.from_numpy(data.hx1.toarray()).float().to(device)
    hx2 = torch.from_numpy(data.hx2.toarray()).float().to(device)
    hy1 = torch.from_numpy(data.hy1.toarray()).float().to(device)
    hy2 = torch.from_numpy(data.hy2.toarray()).float().to(device)

    idx_train = torch.LongTensor(data.idx_train.astype(np.int64)).to(device)
    idx_test = torch.LongTensor(data.idx_test.astype(np.int64)).to(device)
    labels = torch.LongTensor(np.where(data.labels)[1]).to(device)

    gamma = args.gamma
    epochs = args.epochs
    learning_rate = args.learning_rate

    x_n_nodes = X.shape[0]
    y_n_nodes = Y.shape[0]
    pos_weight = float(H_trainX.shape[0] * H_trainX.shape[0] - H_trainX.sum()) / H_trainX.sum()

    model = HCoN(X.shape[1], Y.shape[1], args.dim_hidden, data.n_class)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)



    cost_val = []
    time_list = []
    total_start_time = time.time()

    best_train_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()

        # 记录前向传播时间
        forward_start = time.time()
        recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta)
        forward_pass_time = time.time() - forward_start

        # 记录损失计算时间
        loss_start = time.time()
        loss1 = F.nll_loss(x_output[idx_train], labels[idx_train])
        loss2 = F.binary_cross_entropy_with_logits(recovered, H_trainX, pos_weight=pos_weight)
        loss_train = loss1 + gamma * loss2

        acc_train = accuracy(x_output[idx_train], labels[idx_train])
        loss_calc_time = time.time() - loss_start

        # 记录反向传播时间
        backward_start = time.time()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        backward_pass_time = time.time() - backward_start

        best_train_acc = max(best_train_acc, acc_train)

        # 验证阶段
        with torch.no_grad():
            model.eval()
            val_start = time.time()
            recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta)
            val_loss = F.nll_loss(x_output[idx_test], labels[idx_test])
            test_acc = accuracy(x_output[idx_test], labels[idx_test])

            if test_acc > best_test_acc:
                best_test_acc = test_acc

        total_epoch_time = time.time() - epoch_start_time



    total_time = time.time() - total_start_time

    return best_test_acc.item(), best_train_acc.item(), total_time, []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hypergraph Collaborative Network (HCoN)')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, nargs='?', default='acm_co_I_10%', help="dataset to run")
    setting = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # device = torch.cuda.current_device()
    device='cpu'
    H, X, Y, labels, idx_train_list, idx_test_list, idx_pick = load_data(setting.dataname)

    H_trainX = H.copy()
    Y = np.eye(H.shape[1])
    hx1, hx2 = normalize_sparse_hypergraph_symmetric(H_trainX)
    hy1, hy2 = normalize_sparse_hypergraph_symmetric(H_trainX.transpose())

    dim_hidden = 512
    learning_rate = 0.001
    weight_decay = 0.001
    gamma = 10
    alpha = 0.8
    beta = 0.2

    if setting.dataname == "citeseer":
        dim_hidden = 512
        learning_rate = 0.001
        weight_decay = 0.001
        gamma = 10
        alpha = 0.8
        beta = 0.2

    epochs = 200
    seed = 2021
    early = 100

    # 创建result目录
    import pandas as pd
    import scipy.io as scio

    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    acc_test = []
    train_acc = []
    total_times = []
    for trial_idx, trial in enumerate(idx_pick):
        idx_train = idx_train_list[trial]
        idx_test = idx_test_list[trial]
        data = dotdict()
        args = dotdict()

        data.X = X
        data.Y = Y
        data.H_trainX = H_trainX
        data.hx1 = hx1
        data.hx2 = hx2
        data.hy1 = hy1
        data.hy2 = hy2
        data.labels = labels
        data.idx_train = idx_train
        data.idx_test = idx_test
        data.n_class = labels.shape[1]

        args.dim_hidden = dim_hidden
        args.weight_decay = weight_decay
        args.epochs = epochs
        args.early_stop = early
        args.learning_rate = learning_rate
        args.gamma = gamma
        args.alpha = alpha
        args.beta = beta

        test_acc, train_acc_trial, total_time, time_list = training(data, args, s=seed, trial_id=trial_idx + 1,
                                                                    dataset_name=setting.dataname)
        acc_test.append(test_acc)
        train_acc.append(train_acc_trial)
        total_times.append(total_time)

    # 保存结果到公共result文件夹
    test_accuracy_list = [acc * 100 for acc in acc_test]
    train_accuracy_list = [acc * 100 for acc in train_acc]
    result_path = os.path.join(result_dir, f"HCoN_{setting.dataname}_trials.csv")
    # 保存所有trial的结果
    df = pd.DataFrame({
        'trial': list(range(1, len(test_accuracy_list) + 1)),
        'trial_idx': list(range(len(test_accuracy_list))),
        'test_accuracy': test_accuracy_list,
        'train_accuracy': train_accuracy_list,
    })
    df.to_csv(result_path, index=False)

    # 打印统计信息
    m_test_acc = np.mean(test_accuracy_list)
    s_test_acc = np.std(test_accuracy_list)
    m_train_acc = np.mean(train_accuracy_list)
    s_train_acc = np.std(train_accuracy_list)
    print(
        f"HCoN_{setting.dataname} 测试准确率: {m_test_acc:.4f} ± {s_test_acc:.4f}, 训练准确率: {m_train_acc:.4f} ± {s_train_acc:.4f}")

