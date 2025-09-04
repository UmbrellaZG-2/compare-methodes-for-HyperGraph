import numpy as np
import scipy.sparse as sp
import torch
from LE import transform
import configparser
import scipy.io as scio
import os
import csv

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def catchSplit(dataset):
    cp = configparser.SafeConfigParser()
    cp.read('../config/LE.conf')
    split = []
    train = int(cp[dataset]['id_train'])
    val = int(cp[dataset]['id_val'])
    nodeNum = int(cp[dataset]['nodeNum'])

    # 生成索引
    id_train = np.random.choice(np.arange(nodeNum), train, replace=False)
    remaining = np.array(list(set(np.arange(nodeNum)) - set(id_train)))
    id_val = np.random.choice(remaining, val, replace=False)
    id_test = np.array(list(set(np.arange(nodeNum)) - set(id_train) - set(id_val)))


    return id_train, id_val, id_test

def load_data(dataset_str=None):
    # 使用相对路径获取data文件夹
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # 确保data文件夹存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据文件夹不存在: {data_dir}")
    
    # 扫描data文件夹下的所有mat文件
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    if not mat_files:
        raise FileNotFoundError(f"在{data_dir}中未找到任何.mat文件")
    
    # 如果需要获取所有mat文件路径
    if dataset_str == 'all':
        return [os.path.join(data_dir, f) for f in mat_files]
    
    # 确定要加载的数据集文件
    if dataset_str is None:
        # 如果未指定数据集，使用第一个找到的mat文件
        data_file = mat_files[0]
        dataset_str = data_file[:-4]  # 移除.mat后缀
        print(f"未指定数据集，使用默认数据集: {dataset_str}")
    else:
        # 检查指定的数据集是否存在
        data_file = f"{dataset_str}.mat"
        if data_file not in mat_files:
            # 如果不存在，使用第一个找到的mat文件
            data_file = mat_files[0]
            dataset_str = data_file[:-4]
            print(f"指定的数据集{data_file}不存在，使用默认数据集: {dataset_str}")
    
    # 构建数据文件的相对路径
    data_path = os.path.join(data_dir, data_file)
    print(f"正在加载数据文件: {data_path}")
    import scipy.io as scio
    data_mat = scio.loadmat(data_path)
    # 加载数据
    H = data_mat['H']  # 重命名h为adj
    X = data_mat['X0']  # 重命名X为features
    labels = data_mat['labels']
    idx_train = np.maximum(data_mat['idx_train']-1, 0)
    idx_val = data_mat['idx_test']-1
    # 假设测试集与验证集相同，如果有单独的测试集索引可以修改这里
    idx_test = idx_val.copy()
    
    X = normalize_features(X)
    Y = np.eye(H.shape[1])  # 重命名Y为PvT
    
    return H, Y, X, labels, idx_train, idx_val, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv).dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
