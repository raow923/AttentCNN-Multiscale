"""
数据集构建标准文件
Args:
    1 一个萝卜一个坑
    2 多少萝卜多少坑
-----------------------------------------------------------
Std标准文件-数据集构建标准文件
"""
import random
import numpy as np
from torch.utils.data import Dataset


class train_BCIDataset(Dataset):
    def __init__(self, num_data, train_data, win_train, channel):
        super(train_BCIDataset, self).__init__()

        x_train, y_train = list(range(num_data)), list(range(num_data))

        # 从24组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            y_data =  1  # 获取标签

            time_start = random.randint(1, int(250 + 1 - win_train))  # 随机时间

            x1 = time_start
            x2 = time_start + win_train
            train_data = np.array(train_data)
            x_1 = train_data[:, :, x1:x2]
            x_2 = np.reshape(x_1, (3, channel, win_train))

            x_train[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_train[i] = y_data

        self.data = x_train
        self.label = y_train
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]

