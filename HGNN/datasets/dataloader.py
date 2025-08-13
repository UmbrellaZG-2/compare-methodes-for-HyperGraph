import scipy.io as scio
import numpy as np
import scipy.sparse as sp
import os
import scipy.io as scio
from utils.hypergraph_utils import normalize_features

def load_data(dataset_str=None):
    # 使用相对路径获取data文件夹
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    
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
    h = data_mat.get('H', data_mat.get('h'))
    X = data_mat.get('X0', data_mat.get('X'))
    labels = data_mat.get('labels')
    idx_train_list = data_mat.get('idx_train', data_mat.get('idx_train_list'))
    idx_test_list = data_mat.get('idx_test', data_mat.get('idx_test_list'))
    
    # 检查是否获取到了所有必要的数据
    if h is None or X is None or labels is None or idx_train_list is None or idx_test_list is None:
        raise ValueError(f"数据文件{data_file}中缺少必要的键")
    
    # 调整索引（如果需要）
    if idx_train_list.min() > 0:
        idx_train_list = idx_train_list - 1
    if idx_test_list.min() > 0:
        idx_test_list = idx_test_list - 1
    
    X = normalize_features(X)
    
    return h, X, labels, idx_train_list, idx_test_list