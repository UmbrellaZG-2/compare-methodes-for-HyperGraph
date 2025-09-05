import os
import time
import copy

import pandas as pd
import torch
import torch.optim as optim
import utils.hypergraph_utils as hgut
from utils.hypergraph_utils import accuracy, seed_everything
from models import HGNN
from datasets.dataloader import load_data
import argparse
import numpy as np
from torch import nn

seed_everything(2022)
parser = argparse.ArgumentParser(description='HGNN')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--dataname', type=str, nargs='?',  help="dataname to run")
setting = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.cuda.current_device()


def _main(fts, n_class, idx_train, idx_test, H, lbls):
    model_ft = HGNN(in_ch=fts.shape[1],
                    n_class=n_class,
                    n_hid=128,
                    dropout=0.5)
    model_ft = model_ft.to(device)

    optimizer = optim.Adam(model_ft.parameters(), lr=0.001,
                           weight_decay=5e-4)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100],
                                               gamma=0.9)
    criterion = nn.BCELoss()

    model_ft, result_test, result_train = train_model(model_ft, criterion, optimizer, schedular, idx_train, idx_test, fts, H, lbls,
                                        200, print_freq=100, trial_id=trial+1, dataset_name=setting.dataname)


    return result_test, result_train


def train_model(model, criterion, optimizer, scheduler, idx_train, idx_test, fts, H, lbls, num_epochs=25,
                print_freq=300, trial_id=0, dataset_name=""):
    since = time.time()
    
    # 创建epoch级详细时间记录
    epoch_details = []
    detailed_csv_path = os.path.join(os.path.dirname(__file__), "..", "result", "detailed", f"HGNN_{dataset_name}_trial{trial_id}_epochs.csv")
    os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)
    
    # 写入表头
    pd.DataFrame(columns=['epoch', 'forward_pass_time', 'loss_calc_time', 'backward_pass_time', 
                         'total_epoch_time', 'train_accuracy', 'test_accuracy']).to_csv(detailed_csv_path, index=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}')

        epoch_start_time = time.time()
        forward_pass_time = 0.0
        loss_calc_time = 0.0
        backward_pass_time = 0.0

        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        # 记录前向传播时间
        forward_start = time.time()
        outputs = model(fts, H)
        outputs = torch.sigmoid(outputs)
        forward_pass_time = time.time() - forward_start
        
        # 记录损失计算时间
        loss_start = time.time()
        loss = criterion(outputs[idx_train], lbls[idx_train].type(torch.float))
        train_acc = accuracy(outputs[idx_train].data.cpu().numpy(), lbls[idx_train].data.cpu().numpy())
        loss_calc_time = time.time() - loss_start
        
        # 记录反向传播时间
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        scheduler.step()
        backward_pass_time = time.time() - backward_start

        best_train_acc = max(best_train_acc, train_acc)

        # 验证阶段
        with torch.no_grad():
            model.eval()
            val_start = time.time()
            outputs = model(fts, H)
            outputs = torch.sigmoid(outputs)
            val_loss = criterion(outputs[idx_test], lbls[idx_test].type(torch.float))
            test_acc = accuracy(outputs[idx_test].data.cpu().numpy(), lbls[idx_test].data.cpu().numpy())
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        total_epoch_time = time.time() - epoch_start_time
        
        # 保存epoch详细信息
        epoch_data = {
            'epoch': epoch + 1,
            'forward_pass_time': forward_pass_time,
            'loss_calc_time': loss_calc_time,
            'backward_pass_time': backward_pass_time,
            'total_epoch_time': total_epoch_time,
            'train_accuracy': train_acc * 100,
            'test_accuracy': test_acc * 100
        }
        pd.DataFrame([epoch_data]).to_csv(detailed_csv_path, index=False, mode='a', header=False)

    model.load_state_dict(best_model_wts)
    return model, best_test_acc, best_train_acc


if __name__ == '__main__':

    h, X, labels, idx_train_list, idx_val_list = load_data(setting.dataname)
    
    H = h.toarray()
    fts = X.toarray()
    lbls = labels
    G = hgut.generate_G_from_H(H)
    fts = np.array(fts)

    n_class = lbls.shape[1]

    fts = torch.Tensor(fts).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    G = torch.Tensor(G).to(device)


    test_accuracy_list = []
    train_accuracy_list = []
    save_csv = os.path.join(os.path.dirname(__file__), "..", "result", "HGNN_" + setting.dataname + ".csv")
    lst = ['trial', 'train_acc', 'test_acc']
    # 检查文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(save_csv):
        pd.DataFrame(columns=lst).to_csv(save_csv, index=False)

    for trial in range(1):
        # 开始记录当前trial的时间
        trial_start_time = time.time()

        idx_train = idx_train_list[trial]
        idx_test = idx_val_list[trial]
        
        idx_train = torch.Tensor(idx_train.astype(np.int64)).long().to(device)
        idx_test = torch.Tensor(idx_test.astype(np.int64)).long().to(device)

        test_acc, train_acc = _main(fts, n_class, idx_train, idx_test, G, lbls)
        test_accuracy_list.append(test_acc)
        train_accuracy_list.append(train_acc)

        # 计算当前trial的总时间
        trial_end_time = time.time()
        trial_time = trial_end_time - trial_start_time

        # 将当前trial的数据写入CSV文件
        result_data = {'trial': [trial + 1], 'train_acc': [train_acc * 100], 'test_acc': [test_acc * 100]}  # 准确率转换为百分比
        result = pd.DataFrame(result_data)
        result.to_csv(save_csv, index=False, mode='a', header=False)

        print(f"Trial {trial + 1} test_acc: {test_acc * 100:.2f}%, train_acc: {train_acc * 100:.2f}%, time: {trial_time:.4f} seconds")



