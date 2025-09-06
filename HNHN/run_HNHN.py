import _init_paths
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import time
import copy
import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
import utils
from collections import defaultdict
import pandas as pd
from dataloader import load_data



def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HyperMod(nn.Module):

    def __init__(self, input_dim, vidx, eidx, nv, ne, v_weight, e_weight, args, is_last=False, use_edge_lin=False):
        super(HyperMod, self).__init__()
        self.args = args
        self.eidx = eidx
        self.vidx = vidx
        self.v_weight = v_weight
        self.e_weight = e_weight
        self.nv, self.ne = args.nv, args.ne

        self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.b_v = Parameter(torch.zeros(args.n_hidden))
        self.b_e = Parameter(torch.zeros(args.n_hidden))
        self.is_last_mod = is_last
        self.use_edge_lin = use_edge_lin
        if is_last and self.use_edge_lin:
            self.edge_lin = torch.nn.Linear(args.n_hidden, args.final_edge_dim)

    def forward(self, v, e):

        if self.args.edge_linear:
            ve = torch.matmul(v, self.W_v2e) + self.b_v
        else:
            ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        v_fac = 4 if self.args.predict_edge else 1
        v = v * self.v_weight * v_fac

        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve * self.v_weight)[self.args.ver2edg[:, 0]]
        ve *= self.args.v_reg_weight
        e.scatter_add_(src=ve, index=eidx, dim=0)
        e /= self.args.e_reg_sum
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)

        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev * self.e_weight)[self.args.ver2edg[:, 1]]
        ev_vtx *= args.e_reg_weight
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        v /= args.v_reg_sum
        if not self.is_last_mod:
            v = F.dropout(v, self.args.dropout_p)
        if self.is_last_mod and self.use_edge_lin:
            ev_edge = (ev * torch.exp(self.e_weight) / np.exp(2))[self.args.ver2edg[:, 1]]
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e

    def forward00(self, v, e):
        v = self.lin1(v)
        ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        v = v * self.v_weight

        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve * self.v_weight)[self.args.ver2edg[:, 0]]
        e.scatter_add_(src=ve, index=eidx, dim=0)

        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)
        e = e * self.e_weight
        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev * self.e_weight)[self.args.ver2edg[:, 1]]
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        if self.is_last_mod:
            ev_edge = (ev * torch.exp(self.e_weight) / np.exp(2))[self.args.ver2edg[:, 1]]
            pdb.set_trace()
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e


class Hypergraph(nn.Module):

    def __init__(self, vidx, eidx, nv, ne, v_weight, e_weight, args):
        super(Hypergraph, self).__init__()
        self.args = args
        self.hypermods = []
        is_first = True
        for i in range(args.n_layers):
            is_last = True if i == args.n_layers - 1 else False
            self.hypermods.append(
                HyperMod(args.input_dim if is_first else args.n_hidden, vidx, eidx, nv, ne, v_weight, e_weight, args,
                         is_last=is_last))
            is_first = False

        if args.predict_edge:
            self.edge_lin = torch.nn.Linear(args.input_dim, args.n_hidden)

        self.vtx_lin = torch.nn.Linear(args.input_dim, args.n_hidden)
        self.cls = nn.Linear(args.n_hidden, args.n_cls)

    def to_device(self, device):
        self.to(device)
        for mod in self.hypermods:
            mod.to(device)
        return self

    def all_params(self):
        params = []
        for mod in self.hypermods:
            params.extend(mod.parameters())
        return params

    def forward(self, v, e):
        v = self.vtx_lin(v)
        if self.args.predict_edge:
            e = self.edge_lin(e)
        for mod in self.hypermods:
            v, e = mod(v, e)

        pred = self.cls(v)
        return v, e, pred


class Hypertrain:
    def __init__(self, args):

        self.loss_fn = nn.CrossEntropyLoss()

        self.hypergraph = Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight, args)
        self.optim = optim.Adam(self.hypergraph.all_params(), lr=.01, weight_decay=1e-4)
        self.args = args

    def train(self, v, e, label_idx, labels):
        self.hypergraph = self.hypergraph.to_device(device)
        v_init = v
        e_init = e
        best_err = sys.maxsize
        
        # 计算训练准确率
        train_err, train_acc = self.eval(pred_all, 'train')
        best_train_acc = train_acc
        
        for i in range(self.args.n_epoch):
            args.cur_epoch = i
            v, e, pred_all = self.hypergraph(v_init, e_init)
            pred = pred_all[label_idx.astype(float)]
            loss = self.loss_fn(pred, labels)
            test_err, test_acc = self.eval(pred_all, 'test')
            if test_err < best_err:
                best_err = test_err
            
            # 计算当前epoch的训练准确率
            train_err, train_acc = self.eval(pred_all, 'train')
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        test_err, test_acc = self.eval(pred_all, 'test')
        return pred_all, loss, test_acc, best_train_acc

    def eval(self, all_pred, phase='test'):
        if phase == 'test':
            # 测试集评估
            if self.args.val_idx is None:
                ones = torch.ones(len(all_pred))
                ones[self.args.label_idx] = -1
            else:
                ones = -torch.ones(len(all_pred))
                ones[self.args.val_idx.astype(float)] = 1
        else:
            # 训练集评估
            ones = torch.ones(len(all_pred))
            ones[self.args.label_idx] = 1
            ones[~torch.isin(torch.arange(len(all_pred)), self.args.label_idx)] = -1

        tgt = self.args.all_labels.clone()
        tgt[ones == -1] = -1
        fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = fn(all_pred, tgt)

        tgt = self.args.all_labels[ones > -1]
        pred = torch.argmax(all_pred, -1)[ones > -1]

        acc = torch.eq(pred, tgt).sum().item() / len(tgt)
        return 1 - acc, acc


def train(args, s=616):
    seed_everything(seed=s)
    args.e = torch.zeros(args.ne, args.n_hidden).to(device)
    hypertrain = Hypertrain(args)

    pred_all, loss, test_acc, train_acc = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
    return test_acc, train_acc


def gen_data_cora(args, dataset_name, trail=0):
    h, X, labels, idx_train, idx_test, _ = load_data(dataset_name)

    Y = np.eye(h.shape[1])

    labels = np.array([np.argmax(i) for i in labels])

    num_edge = h.shape[1]

    num_vertex = h.shape[0]

    x = X
    feat_dim = x.shape[-1]

    a, b, c = sp.find(h)
    ver2edg = np.dstack((a, b))

    edge2ver = np.dstack((b, a))

    h = h.todense()
    vertex_weight = np.zeros((h.shape[0]))
    edge_weight = np.zeros((h.shape[1]))

    for i in range(h.shape[0]):
        vertex_weight[i] = np.count_nonzero(h[i])

    for i in range(h.shape[1]):
        edge_weight[i] = np.count_nonzero(h[:, i])

    ver2edg = ver2edg.tolist()
    ver2edg = sum(ver2edg, [])

    edge2ver = edge2ver.tolist()
    edge2ver = sum(edge2ver, [])

    ver2edg = torch.LongTensor(ver2edg).to(device)

    cls_l = list(set(labels))

    cls2int = {k: i for (i, k) in enumerate(cls_l)}
    labels = [cls2int[c] for c in labels]
    args.input_dim = x.shape[-1]
    args.n_hidden = 512
    args.final_edge_dim = 100
    args.n_epoch = 200
    args.ne = num_edge
    args.nv = num_vertex
    ne = args.ne
    nv = args.nv
    args.n_cls = len(cls_l)

    args.all_labels = torch.LongTensor(labels)

    args.label_idx = idx_train[trail]

    args.val_idx = idx_test[trail]

    args.labels = args.all_labels[args.label_idx.astype(float)].to(device)
    args.all_labels = args.all_labels.to(device)
    if isinstance(x, np.ndarray):
        args.v = torch.from_numpy(x.astype(np.float32)).to(device)
    else:
        args.v = torch.from_numpy(np.array(x.astype(np.float32).todense())).to(device)

    args.vidx = ver2edg[:, 0].to(device)
    args.eidx = ver2edg[:, 1].to(device)
    args.ver2edg = ver2edg
    args.v_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in vertex_weight]).unsqueeze(-1).to(device)
    args.e_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in edge_weight]).unsqueeze(-1).to(device)
    assert len(args.v_weight) == nv and len(args.e_weight) == ne

    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(len(ver2edg))
    v_reg_weight = torch.zeros(len(ver2edg))
    use_exp_wt = args.use_exp_wt
    for i, (paper_idx, author_idx) in enumerate(ver2edg.tolist()):
        e_wt = args.e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e * e_wt) if use_exp_wt else e_wt ** args.alpha_e
        e_reg_weight[i] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt)

        v_wt = args.v_weight[paper_idx]
        v_reg_wt = torch.exp(args.alpha_v * v_wt) if use_exp_wt else v_wt ** args.alpha_v
        v_reg_weight[i] = v_reg_wt
        author2sum[author_idx].append(v_reg_wt)
    v_reg_sum = torch.zeros(nv)
    e_reg_sum = torch.zeros(ne)
    for paper_idx, wt_l in paper2sum.items():
        v_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum == 0] = 1
    v_reg_sum[v_reg_sum == 0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    return args


def start_trail(dataset_name, args):
    # 先加载数据以获取trial数量
    h, X, labels, idx_train, idx_test, idx_pick = load_data(dataset_name)
    trials = len(idx_pick)
    test_acc_list = []
    train_acc_list = []
    time_list = []
    
    # 存储每个trial的结果
    trial_results = []
    
    for trial_idx, trial in enumerate(idx_pick):
        args = gen_data_cora(args, dataset_name=dataset_name, trail=trial)
        
        start_time = time.time()
        test_acc, train_acc = train(args)
        end_time = time.time()
        trial_time = end_time - start_time

        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        time_list.append(trial_time)
        
        # 存储当前trial结果
        trial_results.append({
            'trial': trial,
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'time': trial_time
        })

        print(f"Trial {trial_idx + 1} (idx={trial}) test_acc: {test_acc * 100:.2f}%, train_acc: {train_acc * 100:.2f}%, time: {trial_time:.4f} seconds")

    # 计算并打印平均准确率
    avg_test_accuracy = np.mean(test_acc_list)
    avg_train_accuracy = np.mean(train_acc_list)
    avg_time = np.mean(time_list)
    print(f"Average test accuracy over {trials} trials: {avg_test_accuracy * 100:.4f}%")
    print(f"Average train accuracy over {trials} trials: {avg_train_accuracy * 100:.4f}%")
    print(f"HNHN_{dataset_name} 平均时间: {avg_time:.2f}s")
    
    # 保存结果到CSV文件
    import csv
    import os
    
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"HNHN_{dataset_name}_results.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'test_accuracy', 'train_accuracy', 'time'])
        # 写入每个trial的结果
        for result in trial_results:
            writer.writerow([result['trial'], result['test_accuracy'], 
                           result['train_accuracy'], result['time']])
    
    print(f"结果已保存到: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='HNHN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--use_exp_wt', action='store_true', default=False)
    parser.add_argument('--alpha_v', type=float, default=0.1)
    parser.add_argument('--alpha_e', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--edge_linear', action='store_true', default=False)
    parser.add_argument('--predict_edge', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    dataset_name = args.dataset_name
    start_trail(dataset_name, args)
    
