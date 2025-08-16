
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
from collections import defaultdict
import hypergraph
import model.model as hgcn_model
import config.config as hgcn_config
import train as hgnn_train
import time
import sys

import pdb

def train_hgnn(args):
    if args.dataset_name in ['citeseer', 'cora']:
        if args.do_svd:  
            data_path = 'data/citeseer.pt' if args.dataset_name == 'citeseer' else 'data/cora_author_10cls300.pt'  
        else:
            data_path = 'data/citeseer6cls3703.pt' if args.dataset_name == 'citeseer' else 'data/cora_author_10cls1000.pt'

        args = hypergraph.gen_data_cora(args, data_path=data_path, flip_edge_node=False)
    elif args.dataset_name in ['dblp', 'pubmed']:
        data_path = 'data/pubmed_data.pt' if args.dataset_name == 'pubmed' else 'data/dblp_data.pt'
        args = hypergraph.gen_data_dblp(args, data_path=data_path)
    else:
        raise Exception('dataset {} not supported'.format(args.dataset_name))

    X = args.v
    if args.predict_edge:
        labels = torch.cat([args.all_labels, args.edge_classes], 0)
        X = torch.cat([X, args.edge_X], 0)
        n_edge = len(args.edge_X)                
        n_labels = max(1, math.ceil(n_edge*utils.get_label_percent(args.dataset_name)))
        train_idx = torch.from_numpy(np.random.choice(n_edge, size=(n_labels,), replace=False )).to(torch.int64)
        
        all_idx = torch.LongTensor(list(range(len(args.edge_classes))))
        all_idx[train_idx] = -1
        test_idx = all_idx[all_idx > -1]
        train_idx += len(args.all_labels)
        test_idx += len(args.all_labels)
        inc_mx = torch.zeros(len(labels), len(labels))
        e2v = defaultdict(list)
        for i, vidx in enumerate(args.vidx):
            eidx = args.eidx[i]
            e2v[eidx].append(vidx)
        for i in range(len((args.edge_X))):
            e2v[i].append(i+len(args.all_labels))
        for eidx, v_l in e2v.items():
            for vidx1 in v_l:
                for vidx2 in v_l:
                    inc_mx[vidx1, vidx2] = 1

        time0 = time.time()
        acc = hgnn_train._main(X, labels, train_idx, test_idx, inc_mx)
        dur = time.time() - time0
    else:
        labels = args.all_labels
        train_idx = args.label_idx
        all_idx = torch.LongTensor(list(range(len(args.v))))
        all_idx[train_idx] = -1
        test_idx = all_idx[all_idx > -1]
        inc_mx = torch.zeros(len(X), len(X))
        e2v = defaultdict(list)
        for i, vidx in enumerate(args.vidx):
            eidx = args.eidx[i]
            e2v[eidx].append(vidx)

        for eidx, v_l in e2v.items():
            for vidx1 in v_l:
                for vidx2 in v_l:
                    inc_mx[vidx1, vidx2] = 1

        time0 = time.time()
        acc = hgnn_train._main(X, labels, train_idx, test_idx, inc_mx)
        dur = time.time() - time0
    return acc, dur
    
def train_hypergcn(args):

    if args.dataset_name in ['citeseer', 'cora']:
        if args.do_svd:  
            data_path = 'data/citeseer.pt' if args.dataset_name == 'citeseer' else 'data/cora_author_10cls300.pt'
        else:
            data_path = 'data/citeseer6cls3703.pt' if args.dataset_name == 'citeseer' else 'data/cora_author_10cls1000.pt'
        args = hypergraph.gen_data_cora(args, data_path=data_path, flip_edge_node=False)
    elif args.dataset_name in ['dblp', 'pubmed']:
        data_path = 'data/pubmed_data.pt' if args.dataset_name == 'pubmed' else 'data/dblp_data.pt'
        args = hypergraph.gen_data_dblp(args, data_path=data_path)
    else:
        raise Exception('dataset {} not supported'.format(args.dataset_name))
    if args.predict_edge:
        hyp_struct = create_hypergcn_struct(args)
        n_edge = len(args.edge_X)
        n_node = len(args.all_labels)
        labels = torch.cat([args.all_labels, args.edge_classes], 0)
        X = torch.cat([args.v, args.edge_X], 0)
        
        datadict = {'hypergraph':hyp_struct, 'features':args.v, 'labels':labels, 'n':len(args.v)+n_edge}

        n_labels = max(1, math.ceil(n_edge*utils.get_label_percent(args.dataset_name)))
        train_idx = torch.from_numpy(np.random.choice(n_edge, size=(n_labels,), replace=False )).to(torch.int64)
        
        all_idx = torch.LongTensor(list(range(n_edge)))
        all_idx[train_idx] = -1
        test_idx = all_idx[all_idx > -1]
        train_idx += n_node
        test_idx += n_node
        hg_args = hgcn_config.parse()
    else:
        hyp_struct = create_hypergcn_struct(args)
        datadict = {'hypergraph':hyp_struct, 'features':args.v, 'labels':args.all_labels, 'n':len(args.v)}
        train_idx = args.label_idx
        all_idx = torch.LongTensor(list(range(len(args.v))))
        
        all_idx[train_idx] = -1
        test_idx = all_idx[all_idx > -1]
        hg_args = hgcn_config.parse()
    hg_args.n_cls = int(args.all_labels.max() + 1)
    hg_args.fast = args.fast
    save_data = False
    if save_data:
        full_datadict = dict(datadict)
        full_datadict.update({'train_idx': train_idx, 'test_idx': test_idx, 'n_cls': hg_args.n_cls})
        torch.save(full_datadict, '../hypergcn/data/{}_torch.pt'.format(args.dataset_name))
        pdb.set_trace()
        
    
    hg = hgcn_model.initialise(datadict, hg_args)
    time0 = time.time()
    hg = hgcn_model.train(hg, datadict, train_idx, hg_args)
    dur = time.time() - time0
    hg_acc = hgcn_model.test(hg, datadict, test_idx, hg_args)

    print('hg acc ', hg_acc)
    return hg_acc, dur

def create_hypergcn_struct(args):
    paper_author = torch.stack([args.vidx, args.eidx], -1).tolist()
    e2v = defaultdict(set)
    n_node = len(args.all_labels)
    for vidx, eidx in paper_author: 
        e2v[int(eidx+n_node)].add(int(vidx))
    
    if args.predict_edge:
        for i in range(len((args.edge_X))):
            e2v[i].append(i+n_node)

    return dict(e2v) 
    
def gen_data_cora_dep(args):
    data_dict = torch.load('data/cora_author.pt')
    paper_author = torch.LongTensor(data_dict['paper_author'])
    author_paper = torch.LongTensor(data_dict['author_paper'])
    n_author = data_dict['n_author']
    n_paper = data_dict['n_paper']
    classes = data_dict['classes']
    paper_X = data_dict['paper_X']
    paperwt = data_dict['paperwt']
    authorwt = data_dict['authorwt']    
    cls_l = list(set(classes))
    
    cls2int = {k:i for (i, k) in enumerate(cls_l)}
    classes = [cls2int[c] for c in classes]
    args.input_dim = 300
    args.n_hidden = 400
    args.final_edge_dim = 100
    args.n_epoch = 200
    args.ne = n_author
    args.nv = n_paper
    args.n_layers = 1
    ne = args.ne
    nv = args.nv
    args.n_cls = len(cls_l)
    
    n_labels = max(1, int(nv*.052))
    args.all_labels = torch.LongTensor(classes)
    if False:
        n_labels = int(math.ceil(n_labels/args.n_cls)*args.n_cls)
        all_cls_idx = []
        n_label_per_cls = n_labels//args.n_cls
        for i in range(args.n_cls):
            cur_idx = torch.LongTensor(list(range(nv)))[args.all_labels == i]
            rand_idx = torch.from_numpy(np.random.choice(len(cur_idx), size=(n_label_per_cls,), replace=False )).to(torch.int64)
            cur_idx = cur_idx[rand_idx]
            all_cls_idx.append(cur_idx)
        args.label_idx = torch.cat(all_cls_idx, 0)
    else:
        args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False )).to(torch.int64)    

    args.labels = args.all_labels[args.label_idx]
    args.v = torch.from_numpy(paper_X.astype(np.float32))        
    
    args.vidx = paper_author[:, 0]
    args.eidx = paper_author[:, 1]
    args.paper_author = paper_author
    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in paperwt]).unsqueeze(-1)
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in authorwt]).unsqueeze(-1)
    assert len(args.v_weight) == nv and len(args.e_weight) == ne
    train(args)
    
    
if __name__ =='__main__':
    args = utils.parse_args()
    args.fix_seed = True
    method = 'hgnn'
    method = 'hypergcn'
    n_runs = 15
    method = args.method    
    if args.dataset_name == 'cora' and method == 'hypergcn':
        args.dataset_name = 'dblp'
        args.dataset_name = 'citeseer'
    
    if method == 'hypergcn':
        args.fast = False
        if args.dataset_name == 'citeseer':
            n_runs = 15
    print('ARGS {}'.format(args))
    
    if args.fix_seed:
        torch.manual_seed(0)
        np.random.seed(0)
    mean_acc = np.zeros((n_runs))
    time_acc = np.zeros((n_runs))
    for i in range(n_runs):
        if method == 'hgnn':
            acc, dur = train_hgnn(args)
        else:
            acc, dur = train_hypergcn(args)
        sys.stdout.write(' acc {} dur {} '.format(acc, dur))
        mean_acc[i] = acc
        time_acc[i] = dur
    mean_acc_std = np.round(np.std(mean_acc)*100, 2)
    mean_acc_ = np.round(np.mean(mean_acc)*100, 2)
    mean_time = np.round(time_acc.mean(), 4)
    print('acc {}'.format(mean_acc))
    print('mean acc {}+-{} for {} on {} with time {}+-{}'.format(mean_acc_, mean_acc_std, method, args.dataset_name, mean_time, time_acc.std()))
