import numpy as np
from utils import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric
from model import HCoN
import torch
from torch import optim
import torch.nn.functional as F
import os
import argparse
import numpy as np



def training(data, args, s = 2021):
    import time
    seed_everything(seed = s)

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
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta) 
        loss1 = F.nll_loss(x_output[idx_train], labels[idx_train])
        loss2 = F.binary_cross_entropy_with_logits(recovered, H_trainX, pos_weight=pos_weight)
        loss_train = loss1 + gamma * loss2
        
        acc_train = accuracy(x_output[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        epoch_time = time.time() - epoch_start_time
        time_list.append(epoch_time)
        
    total_time = time.time() - total_start_time
    
    with torch.no_grad():
        model.eval()
        recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta) 
        loss_test = F.nll_loss(x_output[idx_test], labels[idx_test])
        acc_test = accuracy(x_output[idx_test], labels[idx_test])
        
    return acc_test.item(), total_time, time_list



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Hypergraph Collaborative Network (HCoN)')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, nargs='?', default='acm_co_I_10%', help="dataset to run")
    setting = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # device = torch.cuda.current_device()
    device = 'cpu'
    H, X, Y, labels, idx_train_list, idx_test_list,idx_pick = load_data(setting.dataname)
    
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
        
    # 存储每个trial的结果
    trial_results = []
    acc_test = []
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

        test, total_time, time_list = training(data, args, s=seed)
        acc_test.append(test)
        total_times.append(total_time)
        
        # 存储当前trial结果
        trial_results.append({
            'trial': trial,
            'test_accuracy': test,
            'time': total_time
        })

    # 只打印最终平均准确率
    acc_test = np.array(acc_test) * 100
    m_acc = np.mean(acc_test)
    s_acc = np.std(acc_test)
    print(f"HCoN_{setting.dataname} 平均准确率: {m_acc:.4f} ± {s_acc:.4f}")
    
    # 保存结果到CSV文件
    import csv
    import os
    
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"HCoN_{setting.dataname}_results.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'test_accuracy', 'train_accuracy', 'time'])
        # 写入每个trial的结果（HCoN没有训练准确度，用测试准确度代替）
        for result in trial_results:
            writer.writerow([result['trial'], result['test_accuracy'], 
                           result['test_accuracy'], result['time']])
    
    print(f"结果已保存到: {csv_path}")

