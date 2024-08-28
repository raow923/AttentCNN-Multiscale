"""
AggtCNN网络训练主逻辑
Args:
    1 时间窗口选择为0.2s;
    2 网络性能评价指标：accuracy, precision, recall and F1-score;
------------------------------------------------------------------------------
Std标准文件-网络训练的标准文件
    1 网络训练有3个阶段：@训练阶段-tra，@验证阶段-val，@测试阶段-tes;
    2 网络训练有三个循环：@受试者循环-sub，@世代循环-epo，@批次循环-bth, @全局global-glo;
    3 需要保存的文件用_log结尾; 中继变量以mid_开头;
    4 Rules:
        #1 变量命名示例1: data_tra_bth;
        #2 变量命名示例1: loss_tra_sub_log;
    5 常见缩写:
        #1 time->tim; #2 window->win; #3 number->num;
        #4 idx->index; #5 predict->pdt; #6 target->tgt;
        #7 middle->mid; #8 iteration->iter;
"""
import os
import csv
import codecs
import torch
import numpy as np
from scipy.io import savemat

from net_std import tCNN
import scipy.io as scio
from scipy import signal
from random import sample
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataloader_std import train_BCIDataset, val_BCIDataset, test_BCIDataset
import time
from read_bdf import read_bdf_data
import random
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.float = float

class SubsetDataset(Dataset):
    def __init__(self, data, labels, indices):
        self.data = [data[i] for i in indices]
        self.labels = [labels[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def baseline_correction(data):
    baseline_mean = np.mean(data, axis=0)  # 计算每个通道的基线平均值
    corrected_data = data - baseline_mean  # 对整个数据减去基线平均值
    return corrected_data

def get_data(wn11, wn21, wn12, wn22, wn13, wn23, path, down_sample):
    data = read_bdf_data(path)  # 读取原始数据

    # 下采样与通道选择
    x_data = data['data'].T
    x_data = np.multiply(x_data, 1000000)
    c = [42, 43, 44, 49, 50, 51, 56, 57, 58]
    train_data = x_data[:, c]
    train_data = baseline_correction(train_data)[::down_sample]
    train_label = np.array(data['event'])[1::2, -1]
    train_start_time = np.array(data['event'])[1::2, 0]

    # @ 滤波1
    channel_data_list1 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn11, wn21], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list1.append(filtedData)
    channel_data_list1 = np.array(channel_data_list1)

    # @ 滤波2
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)

    # @ 滤波3
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)

    return channel_data_list1, channel_data_list2, channel_data_list3, train_label, train_start_time



if __name__ == '__main__':
    # @GPU加速
    start_time = time.time()
    print(torch.cuda.device_count())  # 打印当前设备GPU数量，此笔记本只有1个GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is', device)

    # @训练参数与滤波器设置

    num_epo_tra = 600
    num_data = 5120
    bth_size = 128
    lr = 0.7e-3

    down_sample = 4  # 下采样设置
    fs = 1000 / down_sample  # fs为float类型
    channel = 9  # 选取的通道数

    f_down1 = 6  # 第一个滤波器
    f_up1 = 18
    wn11 = 2 * f_down1 / fs
    wn21 = 2 * f_up1 / fs

    f_down2 = 14  # 第二个滤波器
    f_up2 = 34
    wn12 = 2 * f_down2 / fs
    wn22 = 2 * f_up2 / fs

    f_down3 = 22  # 第三个滤波器
    f_up3 = 50
    wn13 = 2 * f_down3 / fs
    wn23 = 2 * f_up3 / fs

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--tw', type=float, default=0.6, help='Time window (tw) value')
    # 解析参数
    args = parser.parse_args()
    tw = args.tw
    win_data = int(fs * tw)  # 时间窗口对应帧数
    print(f"Using time window: {tw} seconds, which corresponds to {win_data} frames")

    # @网络训练主逻辑
    loss_tra_glo_log, loss_val_glo_log, loss_tes_glo_log = [], [], []
    pdt_tra_glo_log, pdt_val_glo_log, pdt_tes_glo_log = [], [], []
    tgt_tra_glo_log, tgt_val_glo_log, tgt_tes_glo_log = [], [], []
    acc_tra_glo_log, acc_val_glo_log, acc_tes_glo_log = [], [], []
    # @subject循环
    sub = 7      #人数
    exp = 8      #每人实验数
    # 定义数组存储训练文件路径
    tra_paths = []
    for idx_sub in range(1, sub + 1):
    # for idx_sub in range(sub, 0, -1):
        tra_paths.clear() # 清空 tra_paths
        # @数据集设置
        for exp_num in range(1, exp + 1):
            path_sub = f'/home/rao/Data/ssvep01/ssvep01sub0{idx_sub}/EEG_SSVEP0{exp_num}.bdf'
            # path_sub = f'D:\ssvep01\ssvep01sub0{idx_sub}\EEG_SSVEP0{exp_num}.bdf'
            # 将文件路径添加到数组
            tra_paths.append(path_sub)

        # 从文件路径数组中随机选择一个文件路径 做为测试数据路径
        tes_path = random.choice(tra_paths)
        # 从剩余的文件路径数组中移除已选择的路径
        tra_paths.remove(tes_path)
        # 从剩余的文件路径数组中再次随机选择一个文件路径 做为验证数据路径
        val_path = random.choice(tra_paths)
        # 从剩余的文件路径数组中移除已选择的路径 做为训练数据路径
        tra_paths.remove(val_path)

        # @获取滤波后的训练数据、标签和起始时间
        data_tra_list = []
        label_tra_list = []
        start_time_tra_list = []

        # 遍历 tra_paths 数组
        for idx, path in enumerate(tra_paths):
            # 获取数据
            mid_data_tra_1, mid_data_tra_2, mid_data_tra_3, label_tra, start_time_tra = get_data(wn11, wn21, wn12, wn22, wn13, wn23, path, down_sample)

            # 数据聚合, 形状变为4*9*(?)
            data_tra = [mid_data_tra_1, mid_data_tra_2, mid_data_tra_3]

            # 将数据添加到列表中
            data_tra_list.append(data_tra)
            label_tra_list.append(label_tra)
            start_time_tra_list.append(start_time_tra)

        # @获取滤波后的验证数据、标签和起始时间
        mid_data_val_1, mid_data_val_2, mid_data_val_3, label_val, start_time_val = get_data(wn11, wn21, wn12, wn22, wn13, wn23, val_path, down_sample)
        data_val = [mid_data_val_1, mid_data_val_2, mid_data_val_3]
        data_val = np.array(data_val)

        # @获取滤波后的测试数据、标签和起始时间
        mid_data_tes_1, mid_data_tes_2, mid_data_tes_3, label_tes, start_time_tes  = get_data(wn11, wn21, wn12, wn22, wn13, wn23, tes_path, down_sample)
        data_tes = [mid_data_tes_1, mid_data_tes_2, mid_data_tes_3]
        data_tes = np.array(data_tes)

        # @数据集及产生器
        # 初始化空列表以存储数据和标签
        all_data = []
        all_labels = []
        for data_tra, label_tra, start_time_tra in zip(data_tra_list, label_tra_list, start_time_tra_list):
            # 这里执行相同的操作
            dataset_tra = train_BCIDataset(num_data, data_tra, win_data, label_tra, start_time_tra, down_sample, channel)
            # 从 dataset_tra 中提取数据和标签
            for i in range(len(dataset_tra)):
                data, label = dataset_tra[i]
                all_data.append(data)
                all_labels.append(label)

        # 随机抽取 config.num_data 个数据的索引
        selected_indices = random.sample(range(len(all_data)), num_data)

        # 创建 SubsetDataset
        subset_dataset = SubsetDataset(all_data, all_labels, selected_indices)
        # 创建 DataLoader
        gen_tra = DataLoader(subset_dataset, shuffle=True, batch_size=bth_size, num_workers=1, pin_memory=True, drop_last=True)

        dataset_val = val_BCIDataset(num_data, data_val, win_data, label_val, start_time_val, down_sample, channel)
        gen_val = DataLoader(dataset_val, shuffle=True, batch_size=bth_size, num_workers=1, pin_memory=True, drop_last=True)

        dataset_tes = test_BCIDataset(num_data, data_tes, win_data, label_tes, start_time_tes, down_sample, channel)
        gen_tes = DataLoader(dataset_tes, shuffle=True, batch_size=bth_size, num_workers=1, pin_memory=True, drop_last=True)

        # @网络设置
        net = tCNN(win_data)
        net.to(device)
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=0.01)  # 对参数进行正则化weight_decay, 防止过拟合

        loss_tra_sub_log, loss_val_sub_log, loss_tes_sub_log = [], [], []  # 保存的参数
        pdt_tra_sub_log, pdt_val_sub_log, pdt_tes_sub_log = [], [], []
        tgt_tra_sub_log, tgt_val_sub_log, tgt_tes_sub_log = [], [], []
        acc_tra_sub_log, acc_val_sub_log, acc_tes_sub_log = [], [], []

        # @epoch循环
        for epoch in range(num_epo_tra):

            loss_tra_epo_log, loss_val_epo_log, loss_tes_epo_log = [], [], []  # 保存的参数
            pdt_tra_epo_log, pdt_val_epo_log, pdt_tes_epo_log = [], [], []
            tgt_tra_epo_log, tgt_val_epo_log, tgt_tes_epo_log = [], [], []
            acc_tra_epo_log, acc_val_epo_log, acc_tes_epo_log = [], [], []

            epo_size = int(num_data / bth_size)  # 单个世代循环的次数

            # @网络训练及Batch循环
            net.train()
            for iter_tra, bth_tra in enumerate(gen_tra):  # iteration是批次，batch是每批次的数据
                if iter_tra >= epo_size:  # 单世代循环训练退出条件
                    break

                data_tra_bth, tgt_tra_bth = bth_tra[0], bth_tra[1]
                data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)

                optimizer.zero_grad()  # 优化器梯度清零
                output_tra_bth = net(data_tra_bth)
                loss_tra_bth = loss_f(output_tra_bth, tgt_tra_bth.long())  # 算预测值与真实标签之间的损失。
                loss_tra_bth.backward()
                optimizer.step() # 根据计算得到的梯度更新模型参数。
                # 将输出转换为numpy数组，并取最大值的索引，以获得预测结果。
                mid_pdt_tra_bth = np.argmax(output_tra_bth.data.cpu().numpy(), axis=1)
                mid_tgt_tra_bth = tgt_tra_bth.data.cpu().numpy()  # 转化成numpy

                loss_tra_epo_log.append(loss_tra_bth.data.cpu().numpy())  # 将当前批次的损失值添加到损失日志列表中。
                pdt_tra_epo_log.extend(list(mid_pdt_tra_bth))  # 将当前批次的预测结果添加到预测日志列表中。
                tgt_tra_epo_log.extend(list(mid_tgt_tra_bth))  # 将当前批次的目标标签添加到目标标签日志列表中。

            loss_tra_sub_log.append([epoch, np.mean(loss_tra_epo_log)])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_tra_sub_log.append([epoch, pdt_tra_epo_log])
            tgt_tra_sub_log.append([epoch, tgt_tra_epo_log])

            # @网络验证及Batch循环
            net.eval()
            for iter_val, bth_val in enumerate(gen_val):
                if iter_val >= epo_size:
                    break

                data_val_bth, tgt_val_bth = bth_val[0], bth_val[1]
                data_val_bth, tgt_val_bth = data_val_bth.to(device), tgt_val_bth.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()
                    output_val_bth = net(data_val_bth)
                    loss_val_bth = loss_f(output_val_bth, tgt_val_bth.long())

                    mid_pdt_val_bth = np.argmax(output_val_bth.data.cpu().numpy(), axis=1)
                    mid_tgt_val_bth = tgt_val_bth.data.cpu().numpy()  # 转化成numpy

                    loss_val_epo_log.append(loss_val_bth)
                    pdt_val_epo_log.extend(list(mid_pdt_val_bth))
                    tgt_val_epo_log.extend(list(mid_tgt_val_bth))

                    # @计算acc
                    y_true_val_bth = tgt_val_bth.data.cpu().numpy()
                    a_val_bth, b_val_bth = 0, 0
                    for i in range(bth_size):
                        y_pdt_val_bth = np.argmax(output_val_bth.data.cpu().numpy()[i])
                        if y_true_val_bth[i] == y_pdt_val_bth:
                            a_val_bth += 1
                        else:
                            b_val_bth += 1
                    acc_val_bth = a_val_bth / (a_val_bth + b_val_bth)
                    acc_val_epo_log.append(acc_val_bth)

            mid_loss_val_epo = sum(loss_val_epo_log) / epo_size
            mid_acc_val_epo_log = np.mean(acc_val_epo_log)

            loss_val_sub_log.append([epoch, mid_loss_val_epo.data.cpu().numpy()])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_val_sub_log.append([epoch, pdt_val_epo_log])
            tgt_val_sub_log.append([epoch, tgt_val_epo_log])
            acc_val_sub_log.append([epoch, mid_acc_val_epo_log])

            # @网络测试及Batch循环
            net.eval()
            for iter_tes, bth_tes in enumerate(gen_tes):
                if iter_tes >= epo_size:
                    break

                data_tes_bth, tgt_tes_bth = bth_tes[0], bth_tes[1]
                data_tes_bth, tgt_tes_bth = data_tes_bth.to(device), tgt_tes_bth.to(device)

                with torch.no_grad():
                    optimizer.zero_grad()
                    output_tes_bth = net(data_tes_bth)
                    loss_tes_bth = loss_f(output_tes_bth, tgt_tes_bth.long())

                    mid_pdt_tes_bth = np.argmax(output_tes_bth.data.cpu().numpy(), axis=1)
                    mid_tgt_tes_bth = tgt_tes_bth.data.cpu().numpy()  # 转化成numpy

                    loss_tes_epo_log.append(loss_tes_bth)
                    pdt_tes_epo_log.extend(list(mid_pdt_tes_bth))
                    tgt_tes_epo_log.extend(list(mid_tgt_tes_bth))

                    # @计算acc
                    y_true_tes_bth = tgt_tes_bth.data.cpu().numpy()
                    a_tes_bth, b_tes_bth = 0, 0
                    for i in range(bth_size):
                        y_pdt_tes_bth = np.argmax(output_tes_bth.data.cpu().numpy()[i])
                        if y_true_tes_bth[i] == y_pdt_tes_bth:
                            a_tes_bth += 1
                        else:
                            b_tes_bth += 1
                    acc_tes_bth = a_tes_bth / (a_tes_bth + b_tes_bth)
                    acc_tes_epo_log.append(acc_tes_bth)

            mid_loss_tes_epo = sum(loss_tes_epo_log) / epo_size
            mid_acc_tes_epo_log = np.mean(acc_tes_epo_log)

            loss_tes_sub_log.append([epoch, mid_loss_tes_epo.data.cpu().numpy()])  # 先脱离.data,再压到cpu中，再转化成numpy
            pdt_tes_sub_log.append([epoch, pdt_tes_epo_log])
            tgt_tes_sub_log.append([epoch, tgt_tes_epo_log])
            acc_tes_sub_log.append([epoch, mid_acc_tes_epo_log])
            # print('Finish test!')

            end_time = time.time()
            elapsed_time = end_time - start_time
            days = elapsed_time / 86400
            hours = (elapsed_time % 86400) / 3600
            minutes = ((elapsed_time % 86400) % 3600) / 60
            seconds = ((elapsed_time % 86400) % 3600) % 60

            # @每次世代信息反馈
            print('Sub%d, Epo%d, loss_tra%.3f, loss_val is%.3f, loss_tes is%.3f, Acc_val%.3f, Acc_tes%.3f, Time%d days, %dhours, %dminutes, %.3fseconds'
                % (idx_sub, epoch, float(np.mean(loss_tra_epo_log)),
                   np.float(mid_loss_val_epo.data.cpu().numpy()), np.float(mid_loss_tes_epo.data.cpu().numpy()),
                   np.float(mid_acc_val_epo_log), np.float(mid_acc_tes_epo_log), days, hours, minutes, seconds))

        # @训练环节最终保存的数据
        loss_tra_glo_log.append(loss_tra_sub_log)
        pdt_tra_glo_log.append(pdt_tra_sub_log)
        tgt_tra_glo_log.append(tgt_tra_sub_log)

        # @验证环节最终保存的数据
        loss_val_glo_log.append(loss_val_sub_log)
        pdt_val_glo_log.append(pdt_val_sub_log)
        tgt_val_glo_log.append(tgt_val_sub_log)
        acc_val_glo_log.append(acc_val_sub_log)

        # @测试环节最终保存的数据
        loss_tes_glo_log.append(loss_tes_sub_log)
        pdt_tes_glo_log.append(pdt_tes_sub_log)
        tgt_tes_glo_log.append(tgt_tes_sub_log)
        acc_tes_glo_log.append(acc_tes_sub_log)

    # @训练-验证-测试后的文件保存
    path_loss_tra_glo_log = './%.1f/SubAll_loss_tra_glo_log.mat'  % tw
    path_loss_val_glo_log = './%.1f/SubAll_loss_val_glo_log.mat'  % tw
    path_loss_tes_glo_log = './%.1f/SubAll_loss_tes_glo_log.mat'  % tw
    path_acc_val_glo_log = './%.1f/SubAll_acc_val_glo_log.mat'  % tw
    path_acc_tes_glo_log = './%.1f/SubAll_acc_tes_glo_log.mat'  % tw
    path_pdt_tes_glo_log = './%.1f/SubAll_pdt_tes_glo_log.mat'  % tw
    path_tgt_tes_glo_log = './%.1f/SubAll_tgt_tes_glo_log.mat'  % tw


    savemat(path_loss_tra_glo_log, {'tra_loss_glo_log': loss_tra_glo_log})
    savemat(path_loss_val_glo_log, {'val_loss_glo_log': loss_val_sub_log})
    savemat(path_loss_tes_glo_log, {'tes_loss_glo_log': loss_tes_sub_log})
    savemat(path_acc_val_glo_log, {'val_acc_glo_log': acc_val_glo_log})
    savemat(path_acc_tes_glo_log, {'tes_acc_glo_log': acc_tes_glo_log})
    savemat(path_pdt_tes_glo_log, {'tes_pdt_glo_log': pdt_tes_glo_log})
    savemat(path_tgt_tes_glo_log, {'tes_tgt_glo_log': tgt_tes_glo_log})


    print('Train_std is OVER!')
    torch.save(net.state_dict(), 'your_model.pth')

