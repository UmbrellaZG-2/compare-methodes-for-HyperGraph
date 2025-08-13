from model import *
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import os
import scipy.io as scio

# 自定义数据加载函数，自动扫描data文件夹下的mat文件
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
    
    # 加载数据
    data_mat = scio.loadmat(data_path)
    
    # 尝试获取数据，处理可能的键名差异
    H = data_mat.get('H', data_mat.get('h'))
    X = data_mat.get('X0', data_mat.get('X'))
    labels = data_mat.get('labels')
    idx_train = data_mat.get('idx_train', data_mat.get('idx_train_list'))
    idx_test = data_mat.get('idx_test', data_mat.get('idx_test_list'))
    
    # 检查是否获取到了所有必要的数据
    if H is None or X is None or labels is None or idx_train is None or idx_test is None:
        raise ValueError(f"数据文件{data_file}中缺少必要的键")
    
    # 调整索引（如果需要）
    if idx_train.min() > 0:
        idx_train = idx_train - 1
    if idx_test.min() > 0:
        idx_test = idx_test - 1
    
    # 构建数据集字典
    dataset = {
        'features': X,
        'labels': labels,
        'hypergraph': H,
        'train_idx': idx_train,
        'test_idx': idx_test
    }
    
    return dataset, idx_train, idx_test

# 替换原来的data.data导入
# import data.data
# 注意：现在使用上面定义的load_data函数替代data.load

def accuracy(Z, Y):

    return 100 * Z.argmax(1).eq(Y).float().mean().item()


import torch_sparse

def fetch_data(args):
    # 使用自定义的load_data函数加载数据
    dataset, train_idx, test_idx = load_data(args.dataset)
    args.dataset_dict = dataset 

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
   
    # 将scipy.sparse矩阵转换为字典形式的超图表示
    # 假设G是csr_matrix格式
    if sp.issparse(G):
        # 获取超图的边-节点映射
        G = {i: G.indices[G.indptr[i]:G.indptr[i+1]].tolist() for i in range(G.shape[0])}
   
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.cuda(), Y.cuda()
    return X, Y, G, train_idx, test_idx

def initialise(X, Y, G, args, unseen=None):
    """
    initialises model, optimiser, normalises graph, and features
    
    arguments:
    X, Y, G: the entire dataset (with graph, features, labels)
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    returns:
    a tuple with model details (UniGNN, optimiser)    
    """
    
    G = G.copy()
    
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
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
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum'), 'use `mean` or `sum` for first-stage aggregation'
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    V, E = V.cuda(), E.cuda()
    args.degV = degV.cuda()
    args.degE = degE.cuda()
    args.degE2 = degE2.pow(-1.).cuda()


    


    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # UniGNN and optimiser
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



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
