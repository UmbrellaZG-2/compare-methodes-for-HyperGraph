import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import math
import os
import scipy.io as scio
import torch_sparse
import torch.optim as optim, torch.nn.functional as F
from model import UniGNN, HyperGCN


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def load_data(dataset_str=None):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据文件夹不存在: {data_dir}")
    
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    if not mat_files:
        raise FileNotFoundError(f"在{data_dir}中未找到任何.mat文件")
    
    if dataset_str is None:
        data_file = mat_files[0]
        dataset_str = data_file[:-4]
        print(f"未指定数据集，使用默认数据集: {dataset_str}")
    else:
        data_file = f"{dataset_str}.mat"
        if data_file not in mat_files:
            data_file = mat_files[0]
            dataset_str = data_file[:-4]
            print(f"指定的数据集{data_file}不存在，使用默认数据集: {dataset_str}")
    
    data_path = os.path.join(data_dir, data_file)
    print(f"正在加载数据文件: {data_path}")
    data_mat = scio.loadmat(data_path)
    h = data_mat['H']
    X = data_mat['X0']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train']-1
    idx_val_list = data_mat['idx_test']-1
    
    X = normalize_features(X)
    Y = labels
    
    return h, X, Y, labels, idx_train_list, idx_val_list


import numpy as np
from scipy import sparse


def H_to_G(H: sparse.csr_matrix) -> dict:
    """
    将超图关联矩阵 H 转换为字典 G
    Args:
        H: 稀疏关联矩阵 (scipy.sparse.csr_matrix), 形状为 (N, M)
    Returns:
        G: 超图字典, 格式为 {hyperedge_id: [node_id1, node_id2, ...]}
    """
    G = {}

    # 转换为 CSC 格式便于按列访问
    H_csc = H.tocsc()

    for j in range(H_csc.shape[1]):  # 遍历所有超边
        # 获取第 j 列的非零行索引（属于该超边的节点）
        nodes = H_csc.indices[H_csc.indptr[j]:H_csc.indptr[j + 1]]
        G[f"e{j}"] = nodes.tolist()  # 存储为列表

    return G


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



def initialise(X, Y, G, args, unseen=None):
    G = G.copy()

    if unseen is not None:
        unseen = set(unseen)
        for e, vs in G.items():
            G[e] = list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f'self-loop-{v}'] = [v]

    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum')
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1

    V, E = V.cuda(), E.cuda()
    args.degV = degV.cuda()
    args.degE = degE.cuda()
    args.degE2 = degE2.pow(-1.).cuda()

    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    if args.model_name == 'UniGCNII':
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    elif args.model_name == 'HyperGCN':
        args.fast = True
        dataset = args.dataset_dict
        model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset['n'], dataset['hypergraph'], dataset['features'])
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.cuda()

    return model, optimiser

def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()

def normalise(M):
    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)

    return DI.dot(M)
