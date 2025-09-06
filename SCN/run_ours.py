import os
import time
import copy
import torch
import torch.optim as optim
import pprint as pp

from model import ours
import argparse
import scipy
import scipy.io as scio
import numpy as np
import sys
from torch import nn
import random
from utils import seed_everything, load_data_simplices2, load_data_simplices3, accuracy, generate_G_from_H
import csv
import pandas as pd



'''
node classification: python run_ours.py --gpu_id 0 --dataname cora_II --type node_cls
'''


parser = argparse.ArgumentParser(description='ours')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--dataname', type=str, nargs='?', default='cora_II', help="dataname to run")
parser.add_argument('--type', type=str, nargs='?', default='node_cls', help="dataname to run")
setting = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.cuda.current_device()


def train_model(model, criterion, optimizer, idx_train, idx_test, fts_x1, fts_x2, H1, H2, indices, lbls, num_epochs, alpha, trial_id=0, dataset_name=""):
    since = time.time()

    _, labels = torch.max(lbls, 1)
    
    loss_train_list = np.zeros((epochs, 1))
    loss_test_list = np.zeros((epochs, 1))
    acc_train_list = np.zeros((epochs, 1))
    acc_test_list = np.zeros((epochs, 1))
    
    for epoch in range(num_epochs):
        model.train()  
        outputs = model(fts_x1, H1, fts_x2, H2, alpha, indices)
        outputs = outputs[ind_wanted]
        loss_train = criterion(outputs[idx_train], lbls[idx_train].type(torch.float))
        
        _, preds = torch.max(outputs, 1)
        running_corrects = torch.sum(preds[idx_train] == labels[idx_train])
        acc_train = running_corrects.double() / len(idx_train)
        
        # 反向传播
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        loss_train_list[epoch] = loss_train.item()
        acc_train_list[epoch] = acc_train.item()
        
        # 评估阶段
        with torch.no_grad():
            model.eval()
            outputs = model(fts_x1, H1, fts_x2, H2, alpha, indices)
            outputs = outputs[ind_wanted]
            
            loss_test = criterion(outputs[idx_test], lbls[idx_test].type(torch.float))
            _, preds = torch.max(outputs, 1)
            
            running_corrects = torch.sum(preds[idx_test] == labels[idx_test])
            acc_test = running_corrects.double() / len(idx_test)
            
            loss_test_list[epoch] = loss_test.item()
            acc_test_list[epoch] = acc_test.item()


#         loss_val = criterion(outputs[idx_test], lbls[idx_test].type(torch.float))
#         cost_val.append(loss_val.item())
        
#         running_corrects = torch.sum(preds[idx_test] == labels[idx_test])
#         acc_val = running_corrects.double() / len(idx_test)

#         if epoch > 100 and cost_val[-1] > np.mean(cost_val[-(100+1):-1]):
#             print("Early stopping...")
#             break
        
        
    with torch.no_grad():
        model.eval()
        outputs = model(fts_x1, H1, fts_x2, H2, alpha, indices)
        outputs = outputs[ind_wanted]
        
        _, preds = torch.max(outputs, 1)
        running_corrects = torch.sum(preds[idx_test] == labels[idx_test])
        acc_test = running_corrects.double() / len(idx_test)


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'test Acc: {acc_test:4f}\n')

    return acc_test.item(), acc_train_list, acc_test_list, loss_train_list, loss_test_list


def _main(fts_x1, fts_x2, n_class, dim_hidden, dropout, learning_rate, weight_decay, alpha, idx_train, idx_test, H1, H2, indices, lbls, trial_id=0, dataset_name=""):
    
    model_ft = ours(fts_x1.shape[1], dim_hidden, n_class, dropout)
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
#     schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)
    
    criterion = torch.nn.CrossEntropyLoss()
#     criterion = nn.BCELoss()

    acc_test, acc_train_l, acc_test_l, loss_train, loss_test = train_model(model_ft, criterion, optimizer, idx_train, idx_test, fts_x1, fts_x2, H1, H2, indices, lbls, 200, alpha, trial_id, dataset_name)
    
    return acc_test, acc_train_l, acc_test_l, loss_train, loss_test


if __name__ == '__main__':
    
    seed_everything(2021)
    
    # 使用统一的数据路径
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dataset_path = os.path.join(data_dir, f"{setting.dataname}.mat")
    
    H1, H2, X, Y, Z, labels, idx_train_list, idx_val_list, A01, A02, A12, idx_pick = load_data_simplices2(setting.dataname)
    
    X1 = scipy.linalg.block_diag(X, Y, Z)
    X2 = scipy.linalg.block_diag(Z, Y, X)
    n0 = X.shape[0]
    n1 = Y.shape[0]
    n2 = Z.shape[0]
    idx_n0 = np.arange(0, n0)
    idx_n1 = np.arange(n0, n0 + n1)
    idx_n2 = np.arange(n0 + n1, n0 + n1 + n2)
    
    idx_n0 = torch.LongTensor(idx_n0).cuda()
    idx_n1 = torch.LongTensor(idx_n1).cuda()
    idx_n2 = torch.LongTensor(idx_n2).cuda()
    indices = torch.cat((idx_n2, idx_n1), 0)
    indices = torch.cat((indices, idx_n0), 0)
    
    
    
    
    if setting.type == "node_cls":
        ind_wanted = idx_n0
    elif setting.type == "edge_cls":
        ind_wanted = idx_n1
    elif setting.type == "tri_cls":
        ind_wanted = idx_n2
    
    
    
    fts_x1 = X1.copy()
    fts_x2 = X2.copy()

    
    lbls = labels
    n_class = lbls.shape[1]
    
    fts_x1 = torch.Tensor(fts_x1).to(device)
    fts_x2 = torch.Tensor(fts_x2).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    H1 = torch.Tensor(H1.toarray()).to(device)
    H2 = torch.Tensor(H2.toarray()).to(device)
    
    if setting.dataname == "cora_II":
        alpha = 0.8
    else:
        alpha = 0.8  # 为其他数据集设置默认值
    
    
    dim_hidden = 512
    learning_rate = 0.01
    weight_decay = 0.0001
    
    
    epochs = 200
    dropout = 0
    
    # 根据idx_pick的长度初始化数组
    num_trials = len(idx_pick)
    loss_train_list = np.zeros((epochs, num_trials))
    loss_test_list = np.zeros((epochs, num_trials))
    acc_train_list = np.zeros((epochs, num_trials))
    acc_test_list = np.zeros((epochs, num_trials))
    acc_test = []
    for trial_idx, trial in enumerate(idx_pick):
        
        name = f'{dropout}-{dim_hidden}-{weight_decay}-{learning_rate}-{alpha}'
        print(f'ours, {setting.dataname}, GPU: {setting.gpu_id}, Trial: {trial_idx+1} (idx={trial}), Setting: {name} ...')

        idx_train = idx_train_list[trial].astype(int)
        idx_test = idx_val_list[trial].astype(int)

        idx_train = torch.Tensor(idx_train).long().to(device)
        idx_test = torch.Tensor(idx_test).long().to(device)

        acc, acc_train_l, acc_test_l, loss_train, loss_test = _main(fts_x1, fts_x2, n_class, dim_hidden, dropout, learning_rate, weight_decay, alpha, idx_train, idx_test, H1, H2, indices, lbls, trial+1, setting.dataname)

        acc_test.append(acc)
        acc_train_list[0:len(acc_train_l), trial_idx] = acc_train_l.reshape(-1)
        acc_test_list[0:len(acc_test_l), trial_idx] = acc_test_l.reshape(-1)
        loss_train_list[0:len(loss_train), trial_idx] = loss_train.reshape(-1)
        loss_test_list[0:len(loss_test), trial_idx] = loss_test.reshape(-1)


    acc_test = np.array(acc_test) * 100
    m_acc = np.mean(acc_test)
    s_acc = np.std(acc_test)
    print("Test set results:", "accuracy: {:.4f}({:.4f})".format(m_acc, s_acc))

    # 使用统一的结果目录
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存每个trial的汇总结果
    detailed_acc_file = os.path.join(result_dir, f"SCN_{setting.dataname}_trials.csv")
    
    # 创建每个trial的汇总数据
    detailed_data = []
    for trial_idx, trial in enumerate(idx_pick):
        # 获取当前trial的最终训练准确度和最终测试准确度
        final_train_acc = acc_train_list[-1, trial_idx] if len(acc_train_list) > 0 else 0
        final_test_acc = acc_test_list[-1, trial_idx] if len(acc_test_list) > 0 else 0
        
        detailed_data.append({
            'trial': trial_idx + 1,
            'trial_idx': trial,
            'test_accuracy': final_test_acc,
            'train_accuracy': final_train_acc
        })
    
    # 保存每个trial的汇总结果到CSV
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(detailed_acc_file, index=False)
        print(f"Trial results saved to: {detailed_acc_file}")
    
    # 保存汇总结果
    result_file = os.path.join(result_dir, f"SCN_{setting.dataname}.csv")
    results_df = pd.DataFrame({
        'dataset': [setting.dataname],
        'mean_accuracy': [m_acc],
        'std_accuracy': [s_acc],
        'trials': [len(acc_test)]
    })
    results_df.to_csv(result_file, index=False)
    print(f"Summary results saved to: {result_file}")

#     scio.savemat('result/' + setting.dataname + '_ours.mat', {'acc_test': acc_test, 'm_acc': m_acc, 's_acc': s_acc, 
#                                              'acc_train_list': acc_train_list, 
#                                             'acc_test_list': acc_test_list, 
#                                             'loss_train_list': loss_train_list, 
#                                             'loss_test_list': loss_test_list})


    
    
    
    
    
    
    