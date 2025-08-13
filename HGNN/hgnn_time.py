import os
import time
import copy

import pandas as pd
import torch
import torch.optim as optim
import pprint as pp
import utils.hypergraph_utils as hgut
from utils.hypergraph_utils import normalize_features, calculate_metrics, accuracy, seed_everything
from models import HGNN
from config import get_config
# 从datasets.dataloader导入load_data函数
from datasets.dataloader import load_data
import argparse
import numpy as np
import sys
from torch import nn
from datasets.dataloader import load_data

# from ../../utils import normalize_features

seed_everything(2022)
parser = argparse.ArgumentParser(description='HGNN')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--dataname', type=str, nargs='?', required=True, help="dataname to run")
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
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100],
                                               gamma=0.9)
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    model_ft, result_test = train_model(model_ft, criterion, optimizer, schedular, idx_train, idx_test, fts, H, lbls,
                                        200, print_freq=100)

    return result_test


def train_model(model, criterion, optimizer, scheduler, idx_train, idx_test, fts, H, lbls, num_epochs=25,
                print_freq=300):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                train_start = time.time()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, H)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs[idx], lbls[idx].type(torch.float))
                result_train = accuracy(outputs[idx].data.cpu().numpy(), lbls[idx].data.cpu().numpy())
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                optimizer.step()
                scheduler.step()  # 调整到optimizer.step()之后
                train_time = time.time()-train_start

            # deep copy the model
            if phase == 'val':
                test_start = time.time()
                with torch.no_grad():
                    model.eval()
                    outputs = model(fts, H)
                    outputs = torch.sigmoid(outputs)
                    loss_test = criterion(outputs[idx], lbls[idx].type(torch.float))
                    result_test = accuracy(outputs[idx].data.cpu().numpy(), lbls[idx].data.cpu().numpy())
                    test_time = time.time()-test_start

        # 只保存结果，不打印中间过程
        # 只保存最后一个epoch的结果
        if epoch == num_epochs - 1:
            temp = [epoch + 1, train_time, test_time, result_test]
            pd.DataFrame([temp]).to_csv(save_csv, index=False, mode='a+', header=False)


    model.load_state_dict(best_model_wts)
    return model, result_test


if __name__ == '__main__':

    # 使用 dataloader.py 中的 load_data 函数加载数据
    h, X, labels, idx_train_list, idx_val_list = load_data(setting.dataname)
    H = h.toarray()
    # 注意：dataloader.py 没有返回 Y，这里假设 Y 可能不需要或者与 X 相同
    Y = np.eye(H.shape[1])
    # fts = np.eye(H.shape[0])
    fts = X.toarray()
    lbls = labels
    G = hgut.generate_G_from_H(H)
    fts = np.array(fts)

    n_class = lbls.shape[1]

    # print(H.shape[1], type(H))
    # sys.exit()

    # Y = normalize_features(Y)

    # cfg = get_config('config/config.yaml')

    # transform data to device
    fts = torch.Tensor(fts).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    G = torch.Tensor(G).to(device)


    accuracy_list = []
    for trial in range(1):
        idx_train = idx_train_list[trial]
        idx_test = idx_val_list[trial]

        # 转换数据类型为 numpy.int64，以支持 PyTorch 转换
        idx_train = torch.Tensor(idx_train.astype(np.int64)).long().to(device)
        idx_test = torch.Tensor(idx_test.astype(np.int64)).long().to(device)

        # n_class = int(lbls.max()) + 1
        # print(n_class)
        # sys.exit()
        lst = ['epoch', 'train_time', 'test_time']

        # 保存结果到公共result文件夹
        # 保存结果为：项目名+数据集名+时间标识
        save_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "result", "HGNN_" + setting.dataname + "_time.csv"))

        # 修改列名，添加accuracy
        lst = ['epoch', 'train_time', 'test_time', 'accuracy']
        pd.DataFrame(columns=lst).to_csv(save_csv, index=False)
        acc_test = _main(fts, n_class, idx_train, idx_test, G, lbls)
        accuracy_list.append(acc_test)

        # 只打印最终正确率
        print(f"Trial {trial + 1} accuracy: {acc_test}")



