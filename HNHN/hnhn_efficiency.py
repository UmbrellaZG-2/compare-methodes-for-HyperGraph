import pandas as pd

import _init_paths
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import utils
import math
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import sys
import time
import os
import random
import pdb
import scipy.sparse as sp
import scipy.io as scio


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

        if args.edge_linear:
            ve = torch.matmul(v, self.W_v2e) + self.b_v
        else:
            ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
        v_fac = 4 if args.predict_edge else 1
        v = v * self.v_weight * v_fac

        eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve * self.v_weight)[self.args.ver2edg[:, 0]]
        ve *= args.v_reg_weight
        e.scatter_add_(src=ve, index=eidx, dim=0)
        e /= args.e_reg_sum
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)

        vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev * self.e_weight)[self.args.ver2edg[:, 1]]
        ev_vtx *= args.e_reg_weight
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
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
        time_list = []
        for i in range(self.args.n_epoch):
            print(i)
            start = time.time()
            args.cur_epoch = i
            v, e, pred_all = self.hypergraph(v_init, e_init)
            pred = pred_all[label_idx.astype(float)]
            loss = self.loss_fn(pred, labels)
            test_err, acc = self.eval(pred_all)
            if test_err < best_err:
                best_err = test_err
            if args.cur_epoch == 199:
                print('test acc: {}'.format(acc))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            t = time.time() - start
            time_list.append(t)
        print(11)
        e_loss = self.eval(pred_all)
        return pred_all, loss, best_err, acc, time_list

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

        tgt = self.args.all_labels[ones > -1]
        pred = torch.argmax(all_pred, -1)[ones > -1]

        acc = torch.eq(pred, tgt).sum().item() / len(tgt)
        if args.verbose:
            print('acc:{}'.format(acc))
        return 1 - acc, acc


def train(args, s=616):
    seed_everything(seed=s)
    args.e = torch.zeros(args.ne, args.n_hidden).to(device)
    hypertrain = Hypertrain(args)
    print(1)
    pred_all, loss, test_err, acc, time_list = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
    print(2)
    print(time_list)
    return test_err, acc, time_list


def gen_data_cora(args, dataset_name='citeseer', trail=0):
    data_mat = scio.loadmat("../Hypergraph_datasets/{}.mat".format(dataset_name))
    h = data_mat['h']
    X = data_mat['X']
    labels = data_mat['labels']
    idx_train = data_mat['idx_train_list']
    idx_test = data_mat['idx_val_list']

    X = normalize_features(X)

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

    args.label_idx = idx_train[trail] - 1

    args.val_idx = idx_test[trail] - 1

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
    acc_list = np.zeros((1000, 1))

    args.alpha_v = 0.1
    args.alpha_e = 0.1
    lst = ['time_list']

    save_csv = './result/HNHN_' + dataset_name + "_efficiency.csv"

    pd.DataFrame(columns=lst).to_csv(save_csv, index=False)

    for trial in range(1):
        print('trail {}'.format(trial))
        args = gen_data_cora(args, dataset_name=dataset_name, trail=trial)
        test_err, acc, time_list = train(args)

        scio.savemat('Hnhn_efficiency.mat',{'time_list':time_list})
        trials = np.arange(200)
        result = pd.DataFrame({'trial': trials, 'time_list': time_list})
        result.to_csv(save_csv, index=False)

if __name__ == '__main__':
    args = utils.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.cuda.current_device()
    dataset_name = args.dataset_name
    start_trail(dataset_name, args)

