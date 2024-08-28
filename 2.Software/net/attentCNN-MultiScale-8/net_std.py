"""
网络构建标准文件
Args:
    1 输入特定形状数据；
    2 输出特定形状数据；
-----------------------------------------------------------
Std标准文件-网络构建标准文件
"""
import torch
import torch.nn as nn
from block import SELayer
class conv1D_block_(nn.Module):
    def __init__(self, in_channel, out_channel, k_size, stride, drop_rate):
        super(conv1D_block_, self).__init__()
        self.dropout_1 = nn.Dropout(drop_rate)
        self.cnn_cov1d = nn.Conv2d(in_channel, out_channel, k_size, stride, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=0.99, eps=0.001)
        self.elu = nn.ELU(alpha=1)

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.cnn_cov1d(x)
        x = self.bn1(x)
        y = self.elu(x)

        return y


class multi_scale_1D(nn.Module):
    def __init__(self, inc_1, out_channel, first_k, firt_step, drop_out1):
        super(multi_scale_1D, self).__init__()
        self.conv1D_block_1 = conv1D_block_(inc_1, out_channel, first_k, firt_step, drop_out1)

        self.conv1D_block_2 = conv1D_block_(out_channel, out_channel, 32, 1, drop_out1)
        self.conv1D_block_3 = conv1D_block_(out_channel, out_channel, 16, 1, drop_out1)
        self.conv1D_block_4 = conv1D_block_(out_channel, out_channel, 11, 1, drop_out1)

    def forward(self, x):
        x1 = self.conv1D_block_1(x)
        x2 = self.conv1D_block_1(x)
        x3 = self.conv1D_block_1(x)
        x4 = self.conv1D_block_1(x)

        x2_2 = self.conv1D_block_2(x2)
        x3_2 = x3 + x2_2
        x3_3 = self.conv1D_block_3(x3_2)
        x4_3 = x4 + x3_3
        x4_4 = self.conv1D_block_4(x4_3)

        y = x1 + x2_2 + x3_3 + x4_4

        return y

class tCNN(nn.Module):
    def __init__(self, win_train):
        super(tCNN, self).__init__()
        # AFR
        self.se1 = SELayer(3, 2)
        self.se2 = SELayer(3, 2)
        self.se3 = SELayer(3, 2)

        # 第一个卷积，卷积后的数据格式：16@1Xwin_train
        self.conv1 = nn.Conv2d(9, 16, (9, 1))
        self.bn1 = nn.BatchNorm2d(16, momentum=0.99, eps=0.001)
        # self.bn1 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu1 = nn.ELU(alpha=1)
        self.dropout1 = nn.Dropout(0.5)

        self.multi_scale_1D = multi_scale_1D(16, 64, 1, 1, 0.5)

        # 第二个卷积，卷积后的数据格式：16@1X10
        self.conv2 = nn.Conv2d(64, 64, (1, win_train), stride=(5, 5), padding=(0, 24))  # 24也行
        self.bn2 = nn.BatchNorm2d(64, momentum=0.99, eps=0.001)
        # self.bn2 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu2 = nn.ELU(alpha=1)
        self.dropout2 = nn.Dropout(0.5)

        # 第三个卷积，卷积后的数据格式：16@1X6
        self.conv3 = nn.Conv2d(64, 64, (1, 5))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.99, eps=0.001)
        # self.bn3 = nn.BatchNorm2d(16, momentum=0.99)
        self.elu3 = nn.ELU(alpha=1)
        self.dropout3 = nn.Dropout(0.5)

        # 第四个卷积，卷积后的数据格式：32@1X1
        self.conv4 = nn.Conv2d(64, 128, (1, 6))
        self.bn4 = nn.BatchNorm2d(128, momentum=0.99, eps=0.001)
        # self.bn4 = nn.BatchNorm2d(32, momentum=0.99)
        self.elu4 = nn.ELU(alpha=1)
        self.dropout4 = nn.Dropout(0.5)

        # dropout
        self.dropout5 = nn.Dropout(0.5)

        # 全连接
        self.linear = nn.Linear(128, 8)

    def forward(self, x):
        # afr
        x1 = self.se1(x)
        x2 = self.se2(x)
        x3 = self.se3(x)
        x = torch.cat((x1, x2, x3), 1)

        # 第一个卷积, 卷积后的数据格式：16@1X50
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.dropout1(x)

        x = self.multi_scale_1D(x)

        # 第二个卷积, 卷积后的数据格式：16@1X10
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.dropout2(x)

        # 第三个卷积, 卷积后的数据格式：16@1X6
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.dropout3(x)

        # 第四个卷积, 卷积后的数据格式：32@1X1
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu4(x)
        x = self.dropout4(x)

        # 打平
        x = torch.flatten(x, 1, 3)
        x = self.dropout5(x)

        # 全连接
        x = self.linear(x)
        out = x

        return out
