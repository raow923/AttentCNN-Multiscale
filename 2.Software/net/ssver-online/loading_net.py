import torch
import numpy as np
from net_std import tCNN  # 导入你的模型类
import scipy.io as scio
from scipy import signal
from dataloader_std import train_BCIDataset
from torch.utils.data import DataLoader
import serial  # 导入模块
import pyttsx3
from neuracle_api import DataServerThread
import time

def speak(text, rate=200):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  # 设置语速
    engine.say(text)
    engine.runAndWait()
def baseline_correction(data):
    baseline_mean = np.mean(data, axis=0)  # 计算每个通道的基线平均值
    corrected_data = data - baseline_mean  # 对整个数据减去基线平均值
    return corrected_data

def get_data1(wn11, wn21, wn12, wn22, wn13, wn23, path, down_sample):
    # 下采样与通道选择
    x_data = path.T
    x_data = np.multiply(x_data, 10)
    c = [42, 43, 44, 49, 50, 51, 56, 57, 58]
    train_data = x_data[:, c]
    train_data = baseline_correction(train_data)[::down_sample]
    # plt.plot(train_data)
    # plt.xlabel('Index')  # 设置 x 轴标签
    # plt.ylabel('Value')  # 设置 y 轴标签
    # plt.title('Train Data')  # 设置图表标题
    # plt.legend(['Column 42', 'Column 43', 'Column 44', 'Column 49', 'Column 50', 'Column 51', 'Column 56', 'Column 57',
    #             'Column 58'])  # 设置图例
    # plt.show()

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

    return channel_data_list1, channel_data_list2, channel_data_list3

def Port_send(order):
    # 串口初始化
    portx = 'COM7'  # 端口号
    bps = 115200  # 波特率
    timex = 5  # 超时设置
    # ser = serial.Serial(portx, bps, timeout=timex)  # 打开串口,并获得实例对象

    if order == 0:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\xaa\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('橡皮泥!')
    elif order == 1:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\xbb\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('热鸡蛋!')
    elif order == 2:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\xcc\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('苹果!')
    elif order == 3:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\xdd\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('热水!')
    if order == 4:
        x = bytearray(b'\xaa\xbb\x00\xa0\x00\xaa\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('手打开!')
    elif order == 5:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\x00\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('手闭合!')
    elif order == 6:
        x = bytearray(b'\xaa\xbb\x00\xbb\x00\xcc\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('急停关!')
    elif order == 7:
        x = bytearray(b'\xaa\xbb\x00\xaa\x00\xdd\xcc\xdd')
        # ser.write(x)
        # ser.close()
        print('急停开!')
    else:
        print('No Send!')


if __name__ == '__main__':
    win_tim = 0.8  # 时间窗口0.2s

    num_data = 1  # 数据集的大小

    down_sample = 4  # 下采样设置
    fs = 1000 / down_sample  # fs为float类型   每秒的样本数
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

    win_data = int(fs * win_tim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tCNN(win_data)  # 假设你的模型类是 tCNN，需要根据实际情况修改
    model.to(device)
    speak("请选择要抓取的物品", rate=150)
    print('Loading weights into state dict ...')
    time.sleep(2)

    model.load_state_dict(torch.load('mdr.pth', map_location=device))
    model.eval()  # 设置模型为评估模式

    win_data = int(fs * win_tim)  # 时间窗口对应数据帧数

    # 初始化 DataServerThread 线程
    sample_rate = 1000
    t_buffer = 1
    thread_data_server = DataServerThread(sample_rate, t_buffer)
    # 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname='127.0.0.1', port=8712)
    if notconnect:
        raise TypeError("Can't connect JellyFish, Please open the hostport ")
    else:
        # meta包还没解析好就等待
        while not thread_data_server.isReady():
            time.sleep(1)
            continue
        # 启动线程
        thread_data_server.start()
        print('Data server start')
        # 在线数据获取演示：每隔一秒钟，获取数据（数据长度 = time_buffer * srate）
        N, flagstop = 1, False
        while not flagstop:  # get data in one second step
            nUpdate = thread_data_server.GetDataLenCount()
            # print('nUpdate', nUpdate)
            if nUpdate >= sample_rate:
                N += 1
                data = thread_data_server.GetBufferData()
                print('data', data.shape)
                thread_data_server.ResetDataLenCount()
                time.sleep(1)
            if N > 1:
                flagstop = True
        # 结束接收数据
        thread_data_server.stop()

    mid_data_tra_1, mid_data_tra_2, mid_data_tra_3 = get_data1(wn11, wn21, wn12, wn22, wn13, wn23, data, down_sample)
    data_tra = [mid_data_tra_1, mid_data_tra_2, mid_data_tra_3]  # 数据聚合, 形状变为4*9*50
    data_tra = np.array(data_tra)

    dataset_tra = train_BCIDataset(num_data, data_tra, win_data, channel)
    gen_tra = DataLoader(dataset_tra, shuffle=True, batch_size=num_data, num_workers=1, pin_memory=True, drop_last=True)

    for iter_tra, batch in enumerate(gen_tra):
        if iter_tra >= num_data:
            break
        data_test, targets = batch[0], batch[1]
        data_test, targets = data_test.to(device), targets.to(device)
        # 进行预测
        with torch.no_grad():  # 不需要计算梯度
            output = model(data_test)  # 使用转换后的张量进行预测

        predictions = torch.argmax(output, dim=1)  # 获取预测结果

        # 打印预测结果
        print("预测结果:", predictions.item())
        print("真实结果:",  targets.item())
        print("Output:", output)
        # Port_send(targets.item())

    speak("请放下物品", rate=150)
    time.sleep(2)
    Port_send(predictions.item())

    # 初始化 DataServerThread 线程
    sample_rate = 1000
    t_buffer = 1
    thread_data_server = DataServerThread(sample_rate, t_buffer)
    # 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname='127.0.0.1', port=8712)
    if notconnect:
        raise TypeError("Can't connect JellyFish, Please open the hostport ")
    else:
        # meta包还没解析好就等待
        while not thread_data_server.isReady():
            time.sleep(1)
            continue
        # 启动线程
        thread_data_server.start()
        print('Data server start')
        # 在线数据获取演示：每隔一秒钟，获取数据（数据长度 = time_buffer * srate）
        N, flagstop = 1, False
        while not flagstop:  # get data in one second step
            nUpdate = thread_data_server.GetDataLenCount()
            # print('nUpdate', nUpdate)
            if nUpdate >= sample_rate:
                N += 1
                data = thread_data_server.GetBufferData()
                print('data', data.shape)
                thread_data_server.ResetDataLenCount()
                # triggerChan = data[-1, :]
                # # 找到 triggerChan 中所有大于0的元素的位置索引，并将这些索引保存在 idx 中
                # idx = np.argwhere(triggerChan > 0)
                # # print('idx', idx)
                # print('triggerChan', triggerChan[idx])
                time.sleep(1)
            if N > 1:
                flagstop = True
        # 结束接收数据
        thread_data_server.stop()

    mid_data_tra_1, mid_data_tra_2, mid_data_tra_3 \
        = get_data1(wn11, wn21, wn12, wn22, wn13, wn23, data, down_sample)
    data_tra = [mid_data_tra_1, mid_data_tra_2, mid_data_tra_3]  # 数据聚合, 形状变为4*9*50
    data_tra = np.array(data_tra)

    dataset_tra = train_BCIDataset(num_data, data_tra, win_data, 1, 1, down_sample, 15, channel)
    gen_tra = DataLoader(dataset_tra, shuffle=True, batch_size=num_data, num_workers=1,
                         pin_memory=True, drop_last=True)

    for iter_tra, batch in enumerate(gen_tra):
        if iter_tra >= num_data:
            break
        data_test, targets = batch[0], batch[1]
        data_test, targets = data_test.to(device), targets.to(device)
        # 进行预测
        with torch.no_grad():  # 不需要计算梯度
            output = model(data_test)  # 使用转换后的张量进行预测

        predictions = torch.argmax(output, dim=1)  # 获取预测结果

        # 打印预测结果
        print("预测结果:", predictions.item())
        print("真实结果:", targets.item())
        print("Output:", output)
        Port_send(predictions.item())