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
from setting import setting


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
    
    # 创建详细记录文件
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    detailed_csv_path = os.path.join(result_dir, "detailed", f"SCN_{dataset_name}_trial{trial_id}_epochs.csv")
    os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)
    
    epoch_data = []
    
    loss_train_list = np.zeros((epochs, 1))
    loss_test_list = np.zeros((epochs, 1))
    acc_train_list = np.zeros((epochs, 1))
    acc_test_list = np.zeros((epochs, 1))
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 前向传播
        forward_start = time.time()
        model.train()  
        outputs = model(fts_x1, H1, fts_x2, H2, alpha, indices)
        outputs = outputs[ind_wanted]
        loss_train = criterion(outputs[idx_train], lbls[idx_train].type(torch.float))
        forward_time = time.time() - forward_start
        
        _, preds = torch.max(outputs, 1)
        running_corrects = torch.sum(preds[idx_train] == labels[idx_train])
        acc_train = running_corrects.double() / len(idx_train)
        
        # 反向传播
        backward_start = time.time()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        
        loss_train_list[epoch] = loss_train.item()
        acc_train_list[epoch] = acc_train.item()
        
        # 评估阶段
        eval_start = time.time()
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
            eval_time = time.time() - eval_start
        
        # 记录总时间
        total_epoch_time = time.time() - epoch_start_time
        
        # 保存epoch详细数据
        epoch_data.append({
            'epoch': epoch + 1,
            'forward_pass_time': forward_time,
            'loss_calc_time': eval_start - forward_start - forward_time,  # 损失计算时间
            'backward_pass_time': backward_time,
            'total_epoch_time': total_epoch_time,
            'train_accuracy': acc_train.item() * 100,
            'test_accuracy': acc_test.item() * 100
        })
        
    # 不保存详细记录到CSV


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
    
    
    dim_hidden = 512
    learning_rate = 0.01
    weight_decay = 0.0001
    
    
    epochs = 200
    dropout = 0
    
    # 存储每个trial的结果
    trial_results = []
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
        
        # 存储当前trial结果
        trial_results.append({
            'trial': trial,
            'test_accuracy': acc,
            'train_accuracy': np.max(acc_train_l) if len(acc_train_l) > 0 else 0,
            'time': 0  # SCN当前没有时间记录
        })

    acc_test = np.array(acc_test) * 100
    m_acc = np.mean(acc_test)
    s_acc = np.std(acc_test)
    print("Test set results:", "accuracy: {:.4f}({:.4f})".format(m_acc, s_acc))

    # 保存结果到CSV文件
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"SCN_{setting.dataname}_results.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'test_accuracy', 'train_accuracy', 'time'])
        # 写入每个trial的结果
        for result in trial_results:
            writer.writerow([result['trial'], result['test_accuracy'], 
                           result['train_accuracy'], result['time']])
    
    print(f"结果已保存到: {csv_path}")
    
    # 不保存结果到CSV文件
    # print(f"Results completed without CSV saving")

#     scio.savemat('result/' + setting.dataname + '_ours.mat', {'acc_test': acc_test, 'm_acc': m_acc, 's_acc': s_acc, 
#                                              'acc_train_list': acc_train_list, 
#                                             'acc_test_list': acc_test_list, 
#                                             'loss_train_list': loss_train_list, 
#                                             'loss_test_list': loss_test_list})


    
    
    
    
    
    
    