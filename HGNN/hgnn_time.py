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
                                        200, print_freq=100)


    return result_test, result_train


def train_model(model, criterion, optimizer, scheduler, idx_train, idx_test, fts, H, lbls, num_epochs=25,
                print_freq=300):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            idx = idx_train if phase == 'train' else idx_test

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, H)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs[idx], lbls[idx].type(torch.float))
                result_train = accuracy(outputs[idx].data.cpu().numpy(), lbls[idx].data.cpu().numpy())
                if phase == 'train':
                    loss.backward()
                optimizer.step()
                scheduler.step()

            if phase == 'train':
                best_train_acc = max(best_train_acc, result_train)
            else:  # val phase
                with torch.no_grad():
                    model.eval()
                    outputs = model(fts, H)
                    outputs = torch.sigmoid(outputs)
                    loss_test = criterion(outputs[idx], lbls[idx].type(torch.float))
                    result_test = accuracy(outputs[idx].data.cpu().numpy(), lbls[idx].data.cpu().numpy())
                    
                    if result_test > best_test_acc:
                        best_test_acc = result_test
                        best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, best_test_acc, best_train_acc


if __name__ == '__main__':

    h, X, labels, idx_train_list, idx_val_list, idx_pick = load_data(setting.dataname)
    
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

    for trial in range(idx_pick.shape[0]):
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

        print(f"Trial {trial + 1} test_acc: {test_acc * 100:.2f}%, train_acc: {train_acc * 100:.2f}%, time: {trial_time:.4f} seconds")

    # 保存结果到CSV文件
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, f"HGNN_{setting.dataname}_trials.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['trial', 'trial_idx', 'test_accuracy', 'train_accuracy'])
        # 写入每个trial的结果
        for trial_idx, (test_acc, train_acc, trial) in enumerate(zip(test_accuracy_list, train_accuracy_list, idx_pick)):
            writer.writerow([trial_idx + 1, trial, test_acc * 100, train_acc * 100])
    
    print(f"Trial results saved to: {csv_path}")



