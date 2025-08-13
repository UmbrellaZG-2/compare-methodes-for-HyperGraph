import _init_paths
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
# 修正导入错误，使用文件中定义的Hypergraph类
# from hypergraph import Hypergraph
# 注意：Hypergraph类已在当前文件中定义，无需从外部导入
from dataloader import load_data




def normalize_features(mx):
    """Row-normalize sparse matrix"""
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

        if args.edge_linear:
            ve = torch.matmul(v, self.W_v2e) + self.b_v
        else:
            ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        # weigh ve according to how many edges a vertex is connected to
        v_fac = 4 if args.predict_edge else 1
        v = v * self.v_weight * v_fac

        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve * self.v_weight)[self.args.ver2edg[:, 0]]
        ve *= args.v_reg_weight
        e.scatter_add_(src=ve, index=eidx, dim=0)
        e /= args.e_reg_sum
        # e = e*self.e_weight
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)
        # e = e*self.e_weight
        # ev *= self.e_weight
        # v = torch.zeros(self.nv , self.n_hidden)

        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev * self.e_weight)[self.args.ver2edg[:, 1]]
        # ev_vtx = (ev)[self.args.ver2edg[:, 1]]
        ev_vtx *= args.e_reg_weight
        # v = v.clone()
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        # v = v*self.v_weight
        v /= args.v_reg_sum
        if not self.is_last_mod:
            v = F.dropout(v, args.dropout_p)
        if self.is_last_mod and self.use_edge_lin:
            ev_edge = (ev * torch.exp(self.e_weight) / np.exp(2))[self.args.ver2edg[:, 1]]
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e

    def forward00(self, v, e):
        # March normalization
        v = self.lin1(v)
        ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        # weigh ve according to how many edges a vertex is connected to
        # ve *= self.v_weight
        v = v * self.v_weight  # *2

        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve * self.v_weight)[self.args.ver2edg[:, 0]]
        e.scatter_add_(src=ve, index=eidx, dim=0)

        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)
        e = e * self.e_weight
        # ev *= self.e_weight
        # v = torch.zeros(self.nv , self.n_hidden)
        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev * self.e_weight)[self.args.ver2edg[:, 1]]
        # v = v.clone()
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        # v = v*self.v_weight
        if self.is_last_mod:
            ev_edge = (ev * torch.exp(self.e_weight) / np.exp(2))[self.args.ver2edg[:, 1]]
            pdb.set_trace()
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)
        return v, e


class Hypergraph(nn.Module):
    '''
    Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
    One large graph.
    '''

    def __init__(self, vidx, eidx, nv, ne, v_weight, e_weight, args):
        '''
        vidx: idx tensor of elements to select, shape (ne, max_n),
        shifted by 1 to account for 0th elem (which is 0)
        eidx has shape (nv, max n)..
        '''
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
        # insetad of A have vector of indices
        # self.cls = nn.Linear(args.n_hidden+args.final_edge_dim, args.n_cls)
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
        '''
        Take initial embeddings from the select labeled data.
        Return predicted cls.
        '''
        v = self.vtx_lin(v)
        if self.args.predict_edge:
            e = self.edge_lin(e)
        for mod in self.hypermods:
            v, e = mod(v, e)

        pred = self.cls(v)
        return v, e, pred


class Hypertrain:
    def __init__(self, args):

        # cross entropy between predicted and actual labels
        self.loss_fn = nn.CrossEntropyLoss()  # consider logits

        self.hypergraph = Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight, args)
        # optim.Adam([self.P, self.Ly], lr=.4)
        self.optim = optim.Adam(self.hypergraph.all_params(), lr=.01, weight_decay=1e-4)
        # '''
        # milestones = [100 * i for i in range(1, 4)]  # [100, 200, 300]
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.51)
        # '''
        self.args = args

    def train(self, v, e, label_idx, labels):
        self.hypergraph = self.hypergraph.to_device(device)
        # v = v.to(device)
        # e = e.to(device)
        # label_idx = label_idx.to(device)
        # labels = labels.to(device)
        v_init = v
        e_init = e
        best_err = sys.maxsize
        for i in range(self.args.n_epoch):
            args.cur_epoch = i
            v, e, pred_all = self.hypergraph(v_init, e_init)
            pred = pred_all[label_idx.astype(float)]
            loss = self.loss_fn(pred, labels)
            test_err, acc = self.eval(pred_all)
            if test_err < best_err:
                best_err = test_err
            # sys.stdout.write(' loss {} \n'.format(loss))
            # 只在最后一个epoch记录准确率，不打印
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # self.scheduler.step()

        e_loss = self.eval(pred_all)
        return pred_all, loss, best_err, acc

    def eval(self, all_pred):

        if self.args.val_idx is None:
            ones = torch.ones(len(all_pred))
            ones[self.args.label_idx] = -1
        else:
            ones = -torch.ones(len(all_pred))
            ones[self.args.val_idx.astype(float)] = 1

        tgt = self.args.all_labels
        tgt[ones == -1] = -1
        fn = nn.CrossEntropyLoss(ignore_index=-1)
        loss = fn(all_pred, tgt)
        # print(' ~~ eval loss ~~ ', loss)

        tgt = self.args.all_labels[ones > -1]
        # tgt[self.args.label_idx] = -1
        pred = torch.argmax(all_pred, -1)[ones > -1]

        acc = torch.eq(pred, tgt).sum().item() / len(tgt)
        # 不打印评估准确率
        return 1 - acc, acc


def train(args, s=616):
    '''
    args.vidx, args.eidx, args.nv, args.ne, args = s
    args.e_weight = s
    args.v_weight = s
    label_idx, labels = s
    '''
    seed_everything(seed=s)
    args.e = torch.zeros(args.ne, args.n_hidden).to(device)
    hypertrain = Hypertrain(args)

    pred_all, loss, test_err, acc = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
    return test_err, acc


def gen_data_cora(args, dataset_name, trail=0):
    '''
    Retrieve and process data, can be used generically for any dataset with predefined data format, eg cora, citeseer, etc.
    flip_edge_node: whether to flip edge and node in case of relation prediction.
    '''
    # 使用dataloader加载数据
    h, X, labels, idx_train, idx_test = load_data(dataset_name)

    Y = np.eye(h.shape[1])

    # 测试集已经从dataloader中加载

    # 修改labels形式
    labels = np.array([np.argmax(i) for i in labels])
    # print(labels.shape)

    # 边的个数
    num_edge = h.shape[1]
    # print(num_edge)

    # 点的个数
    num_vertex = h.shape[0]

    # feature矩阵
    # print(X)
    x = X
    feat_dim = x.shape[-1]

    # 点到边的矩阵
    a, b, c = sp.find(h)
    # print(type(a)
    ver2edg = np.dstack((a, b))
    # print(ver2edg[0][4])
    # print(h)

    # 边到点的
    edge2ver = np.dstack((b, a))
    # print(edge2ver[0][4])

    # 两个权重问题
    h = h.todense()
    # print(h)
    vertex_weight = np.zeros((h.shape[0]))
    edge_weight = np.zeros((h.shape[1]))

    # print(h)
    for i in range(h.shape[0]):
        vertex_weight[i] = np.count_nonzero(h[i])
    # print(vertex_weight)

    # print(h[:, 0].shape)
    for i in range(h.shape[1]):
        edge_weight[i] = np.count_nonzero(h[:, i])
    # print(edge_weight.shape)

    # 格式转换
    ver2edg = ver2edg.tolist()
    # print(type(ver2edg))
    ver2edg = sum(ver2edg, [])
    # print(ver2edg)

    edge2ver = edge2ver.tolist()
    edge2ver = sum(edge2ver, [])

    ver2edg = torch.LongTensor(ver2edg).to(device)
    # in sparse np array format

    # n_cls = data_dict['n_cls']
    cls_l = list(set(labels))

    cls2int = {k: i for (i, k) in enumerate(cls_l)}
    labels = [cls2int[c] for c in labels]
    args.input_dim = x.shape[-1]  # 300 if args.dataset_name == 'citeseer' else 300
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

    args.labels = args.all_labels[args.label_idx.astype(float)].to(device)  # torch.ones(n_labels, dtype=torch.int64)
    args.all_labels = args.all_labels.to(device)

    if isinstance(x, np.ndarray):
        args.v = torch.from_numpy(x.astype(np.float32)).to(device)
    else:
        args.v = torch.from_numpy(np.array(x.astype(np.float32).todense())).to(device)

    args.vidx = ver2edg[:, 0].to(device)
    args.eidx = ver2edg[:, 1].to(device)
    args.ver2edg = ver2edg
    # pdb.set_trace()
    args.v_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in vertex_weight]).unsqueeze(-1).to(
        device)  # torch.ones((nv, 1)) / 2 #####
    args.e_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in edge_weight]).unsqueeze(-1).to(
        device)  # 1)) / 2 #####torch.ones(ne, 1) / 3
    assert len(args.v_weight) == nv and len(args.e_weight) == ne

    # weights for regularization
    # '''
    paper2sum = defaultdict(list)
    author2sum = defaultdict(list)
    e_reg_weight = torch.zeros(len(ver2edg))  ###
    v_reg_weight = torch.zeros(len(ver2edg))  ###
    # a switch to determine whether to have wt in exponent or base
    use_exp_wt = args.use_exp_wt  # True #False
    for i, (paper_idx, author_idx) in enumerate(ver2edg.tolist()):
        e_wt = args.e_weight[author_idx]
        e_reg_wt = torch.exp(args.alpha_e * e_wt) if use_exp_wt else e_wt ** args.alpha_e
        e_reg_weight[i] = e_reg_wt
        paper2sum[paper_idx].append(e_reg_wt)  ###

        v_wt = args.v_weight[paper_idx]
        v_reg_wt = torch.exp(args.alpha_v * v_wt) if use_exp_wt else v_wt ** args.alpha_v
        v_reg_weight[i] = v_reg_wt
        author2sum[author_idx].append(v_reg_wt)  ###
    # '''
    v_reg_sum = torch.zeros(nv)  ###
    e_reg_sum = torch.zeros(ne)  ###
    for paper_idx, wt_l in paper2sum.items():
        v_reg_sum[paper_idx] = sum(wt_l)
    for author_idx, wt_l in author2sum.items():
        e_reg_sum[author_idx] = sum(wt_l)

    # pdb.set_trace()
    # this is used in denominator only
    e_reg_sum[e_reg_sum == 0] = 1
    v_reg_sum[v_reg_sum == 0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
    return args


def start_trail(dataset_name, args):
    # 保存acc数据
    acc_list = np.zeros((1000, 1))

    args.alpha_v = 0.1
    args.alpha_e = 0.1

    for trial in range(1000):
        args = gen_data_cora(args, dataset_name=dataset_name, trail=trial)
        test_err, acc = train(args)
        acc_list[trial] = acc * 100
        m_acc = np.mean(acc_list)
        # 每100次trial打印一次进度
        if (trial + 1) % 100 == 0:
            print(f'Completed {trial + 1}/1000 trials, current accuracy: {acc:.4f}')
        # print('acc:{}'.format(acc))
        # print(acc_list)
    # 保存结果到公共result文件夹
    # 保存结果为：项目名+数据集名
    result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result', 'HNHN_' + dataset_name + '.csv'))
    # 保存所有参数和结果到result文件夹
    # 创建DataFrame
    df = pd.DataFrame({
        'trial': range(1, len(acc_list) + 1),
        'accuracy': acc_list.flatten(),
        'mean_accuracy': [m_acc] * len(acc_list)
    })
    # 保存到CSV文件
    df.to_csv(result_path, index=False)
    print(f'HNHN_{dataset_name} 平均准确率: {m_acc:.4f}')
    # print(acc_list)


def parse_args():
    parser = argparse.ArgumentParser(description='HNHN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name to run')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.cuda.current_device()
    # device = 'cpu'
    dataset_name = args.dataset_name
    start_trail(dataset_name, args)
    
