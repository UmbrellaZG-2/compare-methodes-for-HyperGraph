import numpy as np
import scipy.sparse as sp
import pickle as pkl
import torch
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import math
from sklearn import metrics
import os
from torch.utils.data import Dataset
import scipy.io as scio
import csv
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, label_ranking_loss, coverage_error, accuracy_score, hamming_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def load_data_simplices2(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    X = data_mat['X0']
    Y = data_mat['X1']
    Z = data_mat['X2']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_test'] - 1
    
    # 获取idx_pick，如果文件中没有则直接报错
    if 'idx_pick' not in data_mat:
        raise ValueError(f"数据文件 data/{dataset_str}.mat 中缺少 'idx_pick' 变量")
    idx_pick = data_mat['idx_pick'].flatten()
    
    X = normalize_features(X).toarray()
    Y = normalize_features(Y).toarray()
    Z = normalize_features(Z).toarray()
    
#     print(X.shape, Y.shape, Z.shape)
#     sys.exit()
    
    H1 = data_mat['H1']
    H2 = data_mat['H2']
    
    A01 = data_mat['A01_ori'].toarray()
    A02 = data_mat['A02_ori'].toarray()
    A12 = data_mat['A12_ori'].toarray()
    
    return H1, H2, X, Y, Z, labels, idx_train_list, idx_val_list, A01, A02, A12, idx_pick

def load_data_simplices2_validation(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    X = data_mat['X0']
    Y = data_mat['X1']
    Z = data_mat['X2']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_val'] - 1
    
    X = normalize_features(X).toarray()
    Y = normalize_features(Y).toarray()
    Z = normalize_features(Z).toarray()
    
#     print(X.shape, Y.shape, Z.shape)
#     sys.exit()
    
    H1 = data_mat['H1']
    H2 = data_mat['H2']
    
    A01 = data_mat['A01_ori'].toarray()
    A02 = data_mat['A02_ori'].toarray()
    A12 = data_mat['A12_ori'].toarray()
    
    return H1, H2, X, Y, Z, labels, idx_train_list, idx_val_list, A01, A02, A12

def load_data_simplices3(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    X0 = data_mat['X0']
    X1 = data_mat['X1']
    X2 = data_mat['X2']
    X3 = data_mat['X3']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_test'] - 1
    
    X0 = normalize_features(X0).toarray()
    X1 = normalize_features(X1).toarray()
    X2 = normalize_features(X2).toarray()
    X3 = normalize_features(X3).toarray()
    
#     print(X.shape, Y.shape, Z.shape)
#     sys.exit()
    
    H1 = data_mat['H1']
    H2 = data_mat['H2']
    
    A01 = data_mat['A01_ori'].toarray()
    A02 = data_mat['A02_ori'].toarray()
    A03 = data_mat['A03_ori'].toarray()
    A12 = data_mat['A12_ori'].toarray()
    A13 = data_mat['A13_ori'].toarray()
    A23 = data_mat['A23_ori'].toarray()
    A = data_mat['H']
    
    return H1, H2, X0, X1, X2, X3, labels, idx_train_list, idx_val_list, A01, A02, A03, A12, A13, A23, A

def load_data_simplices3_validation(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    X0 = data_mat['X0']
    X1 = data_mat['X1']
    X2 = data_mat['X2']
    X3 = data_mat['X3']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_val'] - 1
    
    X0 = normalize_features(X0).toarray()
    X1 = normalize_features(X1).toarray()
    X2 = normalize_features(X2).toarray()
    X3 = normalize_features(X3).toarray()
    
#     print(X.shape, Y.shape, Z.shape)
#     sys.exit()
    
    H1 = data_mat['H1']
    H2 = data_mat['H2']
    
    A01 = data_mat['A01_ori'].toarray()
    A02 = data_mat['A02_ori'].toarray()
    A03 = data_mat['A03_ori'].toarray()
    A12 = data_mat['A12_ori'].toarray()
    A13 = data_mat['A13_ori'].toarray()
    A23 = data_mat['A23_ori'].toarray()
    A = data_mat['H']
    
    return H1, H2, X0, X1, X2, X3, labels, idx_train_list, idx_val_list, A01, A02, A03, A12, A13, A23, A


def load_data_mat(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = data_mat['H']
    X = data_mat['X0']
    Y = data_mat['X1']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_test'] - 1
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, labels, idx_train_list, idx_val_list    

def load_data_mat_validation(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = data_mat['H']
    X = data_mat['X0']
    Y = data_mat['X1']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train'] - 1
    idx_val_list = data_mat['idx_val'] - 1
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, labels, idx_train_list, idx_val_list   
    
    
def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G, DV2 * H * W * invDE

    
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
    
    
    
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 


def ml_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

def hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])
    

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
#     return {
#         'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
#             'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
#             'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
#             'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
#             'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
#             'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
#             'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
#             'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
#             'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
#             'mr': np.all(pred == target, axis=1).mean(),
#             '01Loss': np.any(target != pred, axis=1).mean(),
#             'accuracy': ml_accuracy(target, pred),
#             'hammingLoss': hamming_Loss(target, pred)
#             }

    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'mr': np.all(pred == target, axis=1).mean(),
            '01Loss': np.any(target != pred, axis=1).mean(),
            'accuracy': ml_accuracy(target, pred),
            'hammingLoss': hamming_Loss(target, pred)
            }

def get_results(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    
    return {
        'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
        'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
        'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
        'micro/ap': average_precision_score(target, pred, average='micro'),
        'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
        'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
        'macro/ap': average_precision_score(target, pred, average='macro'),
        'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
        'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
        'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
        'samples/ap': average_precision_score(target, pred, average='samples'),
        'mr': np.all(pred == target, axis=1).mean(),
        '01Loss': np.any(target != pred, axis=1).mean(),
        'rloss': label_ranking_loss(target, pred),
        'cerror': coverage_error(target, pred),
        'accuracy': accuracy_score(target, pred),
        'hammingLoss': hamming_loss(target, pred)
        }

def save_results(result_test, method, save_path, dataname):
    
    micro_precision_list = []
    micro_recall_list = []
    micro_f1_list = []
    micro_ap_list = []
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    macro_ap_list = []
    sample_precision_list = []
    sample_recall_list = []
    sample_f1_list = []
    sample_ap_list = []
    mr_list = []
    loss01_list = []
    rankingLoss_list = []
    coverage_list = []
    accuracy_list = []
    hammingLoss_list = []
    for result in result_test:
        
#         print(result)
        
        micro_p = result['micro/precision']
        micro_r = result['micro/recall']
        micro_f = result['micro/f1']
        micro_ap = result['micro/ap']

        macro_p = result['macro/precision']
        macro_r = result['macro/recall']
        macro_f = result['macro/f1']
        macro_ap = result['macro/ap']

        sample_p = result['samples/precision']
        sample_r = result['samples/recall']
        sample_f = result['samples/f1']
        sample_ap = result['samples/ap']

        mr = result['mr']
        loss01 = result['01Loss']
        rloss = result['rloss']
        cerror = result['cerror']
        accuracy = result['accuracy']
        hammingLoss = result['hammingLoss']
        
        
        micro_precision_list.append(micro_p)
        micro_recall_list.append(micro_r)
        micro_f1_list.append(micro_f)
        micro_ap_list.append(micro_ap)
        macro_precision_list.append(macro_p)
        macro_recall_list.append(macro_r)
        macro_f1_list.append(macro_f)
        macro_ap_list.append(macro_ap)
        sample_precision_list.append(sample_p)
        sample_recall_list.append(sample_r)
        sample_f1_list.append(sample_f)
        sample_ap_list.append(sample_ap)
        mr_list.append(mr)
        loss01_list.append(loss01)
        rankingLoss_list.append(rloss)
        coverage_list.append(cerror)
        accuracy_list.append(accuracy)
        hammingLoss_list.append(hammingLoss)
        
        
        print("precision= {:.4f}".format(micro_p),
              "recall= {:.4f}".format(micro_r),
              "f1= {:.4f}".format(micro_f),
              "mr= {:.4f}".format(mr),
              "loss01= {:.4f}".format(loss01),
              "accuracy= {:.4f}".format(accuracy),
              "hammingLoss= {:.4f} \n".format(hammingLoss))
        
        
    micro_precision_list = np.array(micro_precision_list)
    micro_recall_list = np.array(micro_recall_list)
    micro_f1_list = np.array(micro_f1_list)
    micro_ap_list = np.array(micro_ap_list)
    macro_precision_list = np.array(macro_precision_list)
    macro_recall_list = np.array(macro_recall_list)
    macro_f1_list = np.array(macro_f1_list)
    macro_ap_list = np.array(macro_ap_list)
    sample_precision_list = np.array(sample_precision_list)
    sample_recall_list = np.array(sample_recall_list)
    sample_f1_list = np.array(sample_f1_list)
    sample_ap_list = np.array(sample_ap_list)
    mr_list = np.array(mr_list)
    loss01_list = np.array(loss01_list)
    rankingLoss_list = np.array(rankingLoss_list)
    coverage_list = np.array(coverage_list)
    accuracy_list = np.array(accuracy_list)
    hammingLoss_list = np.array(hammingLoss_list)
    
        
    scio.savemat(save_path + dataname + '_' + method + '.mat', {'micro_precision_list': micro_precision_list,
                                             'micro_recall_list': micro_recall_list, 
                                            'micro_f1_list': micro_f1_list, 
                                                                'micro_ap_list': micro_ap_list, 
                                             'macro_precision_list': macro_precision_list,
                                             'macro_recall_list': macro_recall_list, 
                                            'macro_f1_list': macro_f1_list, 
                                                                'macro_ap_list': macro_ap_list, 
                                                                'sample_precision_list': sample_precision_list,
                                             'sample_recall_list': sample_recall_list, 
                                            'sample_f1_list': sample_f1_list, 
                                                                'sample_ap_list': sample_ap_list, 
                                            'mr_list': mr_list, 
                                            'loss01_list': loss01_list, 
                                                                'rankingLoss_list': rankingLoss_list, 
                                                                'coverage_list': coverage_list, 
                                            'accuracy_list': accuracy_list,
                                            'hammingLoss_list': hammingLoss_list})
    
    
    
def write_results(result, method, save_path, dataname, args):    
    
    micro_p = result['micro/precision']
    micro_r = result['micro/recall']
    micro_f = result['micro/f1']
    micro_ap = result['micro/ap']

    macro_p = result['macro/precision']
    macro_r = result['macro/recall']
    macro_f = result['macro/f1']
    macro_ap = result['macro/ap']

    sample_p = result['samples/precision']
    sample_r = result['samples/recall']
    sample_f = result['samples/f1']
    sample_ap = result['samples/ap']

    mr = result['mr']
    loss01 = result['01Loss']
    rloss = result['rloss']
    cerror = result['cerror']
    accuracy = result['accuracy']
    hammingLoss = result['hammingLoss']



    with open(save_path + "new_" + dataname + "_" + method + ".csv", "a+") as csvfile: 
        
        if method == "HGNN":
            result = (args.dropout, args.dim_hidden, args.learning_rate, args.weight_decay, micro_p, micro_r, micro_f, micro_ap, macro_p, macro_r, macro_f, macro_ap, sample_p, sample_r, sample_f, sample_ap, mr, loss01, rloss, cerror, accuracy, hammingLoss)
            header = ["dropout", "dim_hidden", "learning_rate", "weight_decay", 'micro_p', 'micro_r', 'micro_f', 'micro_ap', 'macro_p', 'macro_r', 'macro_f', 'macro_ap', 'sample_p', 'sample_r', 'sample_f', 'sample_ap', 'mr', '01Loss', 'rankingLoss', 'Coverage', 'accuracy', 'hammingLoss']
            
        elif method == "HCoN" or method == "ours":
            result = (args.dropout, args.dim_hidden, args.learning_rate, args.weight_decay, args.alpha, micro_p, micro_r, micro_f, micro_ap, macro_p, macro_r, macro_f, macro_ap, sample_p, sample_r, sample_f, sample_ap, mr, loss01, rloss, cerror, accuracy, hammingLoss)
            header = ["dropout", "dim_hidden", "learning_rate", "weight_decay", "alpha", 'micro_p', 'micro_r', 'micro_f', 'micro_ap', 'macro_p', 'macro_r', 'macro_f', 'macro_ap', 'sample_p', 'sample_r', 'sample_f', 'sample_ap', 'mr', '01Loss', 'rankingLoss', 'Coverage', 'accuracy', 'hammingLoss']

        writer = csv.writer(csvfile)
        with open(save_path + "new_" + dataname + "_" + method + ".csv", "r") as csvfile: 
            reader = csv.reader(csvfile)

            if not [row for row in reader]:
                writer.writerow(header)
                writer.writerow(result)

            else:
                writer.writerow(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

