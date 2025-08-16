import scipy.io as scio
import numpy as np
import scipy.sparse as sp
import os
import scipy.io as scio

def load_data(dataset_str):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "{}.mat")).format(dataset_str)
    data_mat = scio.loadmat(data_path)
    h = data_mat['H']
    X = data_mat['X0']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train']-1
    idx_test_list = data_mat['idx_test']-1

    X = normalize_features(X)

    return h, X, labels, idx_train_list, idx_test_list

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