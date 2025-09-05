import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import math
import os
import scipy.io as scio
    

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
    Y = np.eye(h.shape[1])
    
    return h, X, Y, labels, idx_train_list, idx_val_list


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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 


def normalize_sparse_hypergraph_symmetric(H):
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)
    
    colsum = np.array(H.sum(0))
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)
    
    Omega = sp.eye(B.shape[0])

    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2

        
        