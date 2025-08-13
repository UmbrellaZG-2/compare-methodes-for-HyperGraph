
'''
Utilities functions for the framework.
'''
import pandas as pd
import numpy as np
import os
import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import pickle
import warnings
import sklearn.metrics
warnings.filterwarnings('ignore')

import pdb
import torch

#torch.set_default_tensor_type('torch.DoubleTensor')
#torch_dtype = torch.float64 #torch.float32

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_G_from_H(H, variable_weight=False):
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
    # invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE = np.diag(np.power(DE, -1))
    # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2 = np.diag(np.power(DV, -0.5))
    H = np.mat(H)
    HT = H.T
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        G = DV2_H * invDE_HT_DV2
    else:
        G = DV2 * H * invDE * HT * DV2
    return G.A

lres_dir = 'results'
data_dir = 'data'
#device = 'cpu' #set to CPU here if checking timing for fair timing comparison

def parse_args():
    parser = argparse.ArgumentParser()
    #    
    parser.add_argument("--verbose", dest='verbose', action='store_const', default=False, const=True, help='Print out verbose info during optimization')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--do_svd", action='store_const', default=False, const=True, help='use svd')
    parser.add_argument("--method", type=str, default='hypergcn', help='which baseline method')
    parser.add_argument("--kfold", default=1, type=int, help='for k-fold cross validation')
    parser.add_argument("--predict_edge", action='store_const', default=False, const=True, help='whether to predict edges')
    parser.add_argument("--edge_linear", action='store_const', default=False, const=True, help='linerity')
    parser.add_argument("--alpha_e", default=0.1, type=float, help='alpha')
    parser.add_argument("--alpha_v", default=0.1, type=float, help='alpha')
    parser.add_argument("--dropout_p", default=0, type=float, help='dropout')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--dataset_name", type=str, default='cora_cocitation_1000', help='dataset name')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")

    opt = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # device = torch.cuda.current_device()
    # device = 'cpu'

    return opt

def readlines(path):
    with open(path, 'r') as f:
        return f.readlines()
    

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    else:
        raise Exception('dataset not supported')
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
