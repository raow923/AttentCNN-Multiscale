# """
# This is the Standard Method Analysis script
# -----------------------------------------------------------------
# 方法@1： 计算多个人的平均准确性 -> acc_mean_std_multiple_sub_ave
# 方法@2： 拼接所有人的最优预测结果 -> con_allPdt_with_best_acc
# 方法@3： 拼接所有人对应@2的真实标签 -> con_allTgt_with_best_acc
# 方法@4： 输出训练的Loss-Epoch演示图 -> loss_tra_plot_multiple_sub_ave
# 方法@5： 输出报告表 -> classification_report_diy
# 方法@6： 获取混淆矩阵 -> getCMatrix
# -----------------------------------------------------------------
# """
# import scipy.io as scio
# import numpy as np
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from matplotlib import rcParams, font_manager
#
#
#
# # @acc_mean_std_tes
# def acc_mean_std_multiple_sub_ave_tes(path, num_sub):
#     """
#     目的：计算多个人的平均准确性
#     步骤：
#     #1 找出单个人的最优准确性
#     #2 求均值即可
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#     acc_data_tmp = data['tes_acc_glo_log']
#
#     # num_sub = acc_data_tmp.shape[0]
#     num_sub = num_sub
#     acc_max_sub = []
#     acc_max_epo_idx = []
#     for idx_sub in range(num_sub):
#         acc_data_tmp2 = acc_data_tmp[idx_sub][:, 1]
#         acc_max_sub_tmp = np.max(acc_data_tmp2)
#         acc_max_sub.append(acc_max_sub_tmp)
#         acc_max_epo_idx_tmp = np.argmax(acc_data_tmp2)
#         acc_max_epo_idx.append(acc_max_epo_idx_tmp)
#     # 计算均值和标准差
#     acc_mean = np.mean(acc_max_sub)
#     acc_std = np.std(acc_max_sub)
#     # 所有参与者的平均准确性、准确性标准差、每个参与者的最佳准确性对应的索引
#     return acc_mean, acc_std, acc_max_epo_idx
#
#
# # @acc_mean_std_tes
# def acc_mean_std_multiple_sub_ave_val(path, num_sub):
#     """
#     目的：计算多个人的平均准确性
#     步骤：
#     #1 找出单个人的最优准确性
#     #2 求均值即可
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#     acc_data_tmp = data['val_acc_glo_log']
#
#     # num_sub = acc_data_tmp.shape[0]
#     num_sub = num_sub
#     acc_max_sub = []
#     acc_max_epo_idx = []
#     for idx_sub in range(num_sub):
#         acc_data_tmp2 = acc_data_tmp[idx_sub][:, 1]
#         acc_max_sub_tmp = np.max(acc_data_tmp2)
#         acc_max_sub.append(acc_max_sub_tmp)
#         acc_max_epo_idx_tmp = np.argmax(acc_data_tmp2)
#         acc_max_epo_idx.append(acc_max_epo_idx_tmp)
#     acc_mean = np.mean(acc_max_sub)
#     acc_std = np.std(acc_max_sub)
#
#     return acc_mean, acc_std, acc_max_epo_idx
#
#
# # @con_allPdt
# def con_allPdt_with_best_acc(path, idx_list):
#     """
#     目的：拼接所有人的最优预测结果
#     步骤：
#     #1 找出单个人的最优预测结果
#     #2 把他们拼接在一起
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#     data_tmp = data['tes_pdt_glo_log']
#     all_sub_pdt = []
#     for idx_sub in range(data_tmp.shape[0]):
#         tmp2 = idx_list[idx_sub]
#         sub_pdt_tmp = data_tmp[idx_sub][tmp2, 1]
#         all_sub_pdt.extend(sub_pdt_tmp)
#
#     return all_sub_pdt
#
#
# # @con_allTgt
# def con_allTgt_with_best_acc(path, idx_list):
#     """
#     # 目的：拼接所有人对应@2的真实标签
#     # 步骤：#1 找出单个人对应最优Epoch的真实标签
#     #     #2 把他们拼接在一起
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#     data_tmp = data['tes_tgt_glo_log']
#     all_sub_tgt = []
#     for idx_sub in range(data_tmp.shape[0]):
#         tmp = idx_list[idx_sub]
#         sub_tgt_tmp = data_tmp[idx_sub][tmp, 1]
#         all_sub_tgt.extend(sub_tgt_tmp)
#
#     return all_sub_tgt
#
#
# # @loss_tra_plot
# def loss_tra_plot_multiple_sub_ave(path):
#     """
#     输出训练的Loss-Epoch演示图
#     步骤：将所有受试者对应的Epoch损失求和再去平均
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#     data_tmp = data['tra_loss_glo_log']
#     loss_sum = np.zeros(data_tmp.shape[1])
#     for idx_sub in range(data_tmp.shape[0]):
#         tmp2 = np.array(data_tmp[idx_sub][:, 1])
#         loss_sum += tmp2
#     loss_ave = loss_sum / data_tmp.shape[0]
#
#     return loss_ave
#
#
# # @loss_tra_plot
# def classification_report_diy(y_true, y_pdt, classes):
#     """
#     y_true和y_pdt输出报告表
#     """
#     report = classification_report(y_true, y_pdt, target_names=classes, output_dict=True)
#
#     return report
#
#
# # @获取混淆矩阵
# def getCMatrix(y_true, y_pdt):
#     cm = confusion_matrix(y_true, y_pdt)
#     # plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
#     return cm
#
# # @根据混淆矩阵绘制图像
# def draw_pic_from_CMatrix(classes, CMatrix):
#     """
#     根据混淆矩阵绘图
#     参考资料：
#     https://blog.csdn.net/Mluoo/article/details/125215306
#     """
#     classes = classes
#     confusion_matrix = CMatrix
#
#     # @计算百分比
#     proportion = []
#     length = len(confusion_matrix)  # 矩阵行数, 即有多少类
#     print("矩阵行数:", length)
#     for i in confusion_matrix:
#         for j in i:
#             tmp = j / np.sum(i)
#             proportion.append(tmp)
#     # print(np.sum(confusion_matrix[0]))
#     # print(proportion)
#
#     # @转化成百分比
#     pshow = []
#     for i in proportion:
#         pt = "%.2f%%" % (i * 100)
#         pshow.append(pt)
#
#     # @转化成length*length形状
#     proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
#     # print(proportion)
#     pshow = np.array(pshow).reshape(length, length)
#     # print(pshow)
#
#     # @画图：
#     # @设置字体
#     font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Times New Roman 字体路径
#     font_prop = font_manager.FontProperties(fname=font_path)
#
#     # @显示出矩阵
#     plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.YlOrBr)
#     # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
#     # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
#     # plt.title('confusion_matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, fontsize=12, fontproperties=font_prop)
#     plt.yticks(tick_marks, classes, fontsize=12, fontproperties=font_prop)
#
#     iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
#     for i, j in iters:
#         if (i == j):
#             plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
#                      weight=5, fontproperties=font_prop)  # 显示对应的数字
#             plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white', fontproperties=font_prop)
#         else:
#             plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, fontproperties=font_prop)  # 显示对应的数字
#             plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, fontproperties=font_prop)
#
#     plt.ylabel('True label', fontsize=16, fontproperties=font_prop)
#     plt.xlabel('Predict label', fontsize=16, fontproperties=font_prop)
#     plt.tight_layout()  # 自动调整间距
#     plt.show()
    # plt.savefig('混淆矩阵.png')
#
# if __name__ == '__main__':
#     # 数据路径
#     path_tes_acc_glo_log = './logs/SubAll_acc_tes_glo_log.mat'
#     path_tes_pdt_glo_log = './logs/SubAll_pdt_tes_glo_log.mat'
#     path_tes_tgt_glo_log = './logs/SubAll_tgt_tes_glo_log.mat'
#     path_tra_loss_glo_log = './logs/SubAll_loss_tra_glo_log.mat'
#
#     acc_mean, acc_std, acc_max_epo_idx = acc_mean_std_multiple_sub_ave_tes(path_tes_acc_glo_log, 54)
#     all_sub_tgt = con_allTgt_with_best_acc(path_tes_tgt_glo_log, acc_max_epo_idx)
#     all_sub_pdt = con_allPdt_with_best_acc(path_tes_pdt_glo_log, acc_max_epo_idx)
#     loss_ave = loss_tra_plot_multiple_sub_ave(path_tra_loss_glo_log)
#
#     all_sub_tgt_tmp = np.array(all_sub_tgt).flatten()  # 将N维度数组转成一维数组
#     all_sub_pdt_tmp = np.array(all_sub_pdt).flatten()
#
#     classes = ['1', '2', '3', '4']  # 类别
#     # CReport = classification_report_diy(all_sub_tgt_tmp, all_sub_pdt_tmp, classes)
#
#     Cmatrix_tmp = getCMatrix(all_sub_tgt_tmp, all_sub_pdt_tmp)
#     draw_pic_from_CMatrix(classes, Cmatrix_tmp)
#     print("MATraditional Is OK!")
#


"""
This is the Standard Method Analysis script
-----------------------------------------------------------------
方法@1： 计算多个人的平均准确性 -> acc_mean_std_multiple_sub_ave
方法@2： 拼接所有人的最优预测结果 -> con_allPdt_with_best_acc
方法@3： 拼接所有人对应@2的真实标签 -> con_allTgt_with_best_acc
方法@4： 输出训练的Loss-Epoch演示图 -> loss_tra_plot_multiple_sub_ave
方法@5： 输出报告表 -> classification_report_diy
方法@6： 获取混淆矩阵 -> getCMatrix
-----------------------------------------------------------------
"""
import scipy.io as scio
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import font_manager


# @acc_mean_std_tes
def acc_mean_std_multiple_sub_ave_tes(num_sub):
    """
    目的：计算多个人的平均准确性
    步骤：
    #1 找出单个人的最优准确性
    #2 求均值即可
    """
    acc_max_sub = []
    acc_max_epo_idx = []
    for idx in range(num_sub):
        path = './logs/Sub%d_tes_acc_sub_log.mat' % (idx + 1)
        data = scio.loadmat(path)  # 读取原始数据
        acc_data_tmp = data['tes_acc_sub_log']
        acc_data_tmp2 = acc_data_tmp[:, 1]
        acc_max_sub_tmp = np.max(acc_data_tmp2)
        acc_max_sub.append(acc_max_sub_tmp)
        acc_max_epo_idx_tmp = np.argmax(acc_data_tmp2)
        acc_max_epo_idx.append(acc_max_epo_idx_tmp)

    acc_mean = np.mean(acc_max_sub)
    acc_std = np.std(acc_max_sub)
    acc_sem = acc_std / np.sqrt(7)

    return acc_mean, acc_std, acc_sem, acc_max_epo_idx


# @acc_mean_std_val
def acc_mean_std_multiple_sub_ave_val(num_sub):
    """
    目的：计算多个人的平均准确性
    步骤：
    #1 找出单个人的最优准确性
    #2 求均值即可
    """
    acc_max_sub = []
    acc_max_epo_idx = []
    for idx in range(num_sub):
        path = './logs/Sub%d_val_acc_sub_log.mat' % (idx + 1)
        data = scio.loadmat(path)  # 读取原始数据
        acc_data_tmp = data['val_acc_sub_log']
        acc_data_tmp2 = acc_data_tmp[:, 1]
        acc_max_sub_tmp = np.max(acc_data_tmp2)
        acc_max_sub.append(acc_max_sub_tmp)
        acc_max_epo_idx_tmp = np.argmax(acc_data_tmp2)
        acc_max_epo_idx.append(acc_max_epo_idx_tmp)

    acc_mean = np.mean(acc_max_sub)
    acc_std = np.std(acc_max_sub)

    return acc_mean, acc_std, acc_max_epo_idx


# @con_allPdt_tes
def con_allPdt_with_best_acc_tes(idx_list):
    """
    目的：拼接所有人的最优预测结果
    步骤：
    #1 找出单个人的最优预测结果
    #2 把他们拼接在一起
    """
    all_sub_pdt = []
    for idx in range(num_sub):
        path = './logs/Sub%d_tes_pdt_sub_log.mat' % (idx + 1)
        data = scio.loadmat(path)  # 读取原始数据
        data_tmp = data['tes_pdt_sub_log']
        tmp2 = idx_list[idx]
        sub_pdt_tmp = data_tmp[tmp2, 1]
        all_sub_pdt.extend(sub_pdt_tmp)
    return all_sub_pdt


# @con_allTgt_tes
def con_allTgt_with_best_acc_tes(idx_list):
    """
    目的：拼接所有人的最优预测结果
    步骤：
    #1 找出单个人的最优预测结果
    #2 把他们拼接在一起
    """
    all_sub_tgt = []
    for idx in range(num_sub):
        path = './logs/Sub%d_tes_tgt_sub_log.mat' % (idx + 1)
        data = scio.loadmat(path)  # 读取原始数据
        data_tmp = data['tes_tgt_sub_log']
        tmp2 = idx_list[idx]
        sub_tgt_tmp = data_tmp[tmp2, 1]
        all_sub_tgt.extend(sub_tgt_tmp)
    return all_sub_tgt


# @loss_tra_plot
def loss_tra_plot_multiple_sub_ave(num_sub, num_epo):
    """
    输出训练的Loss-Epoch演示图
    步骤：将所有受试者对应的Epoch损失求和再去平均
    """
    loss_sum = np.zeros(num_epo)
    for idx in range(num_sub):
        path = './logs/Sub%d_tra_loss_sub_log.mat' % (idx + 1)
        data = scio.loadmat(path)  # 读取原始数据
        data_tmp = data['tra_loss_sub_log']
        tmp2 = np.array(data_tmp[:, 1])
        loss_sum += tmp2
    loss_ave = loss_sum / num_sub

    return loss_ave


# @loss_tra_plot
def classification_report_diy(y_true, y_pdt, classes):
    """
    y_true和y_pdt输出报告表
    """
    report = classification_report(y_true, y_pdt, target_names=classes, output_dict=True)

    return report


# @获取混淆矩阵
def getCMatrix(y_true, y_pdt):
    cm = confusion_matrix(y_true, y_pdt)
    # plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
    return cm



# @根据混淆矩阵绘制图像
def draw_pic_from_CMatrix(classes, CMatrix):
    """
    根据混淆矩阵绘图
    参考资料：
    https://blog.csdn.net/Mluoo/article/details/125215306
    """
    classes = classes
    confusion_matrix = CMatrix

    # @计算百分比
    proportion = []
    length = len(confusion_matrix)  # 矩阵行数, 即有多少类
    print("矩阵行数:", length)
    for i in confusion_matrix:
        for j in i:
            tmp = j / np.sum(i)
            proportion.append(tmp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)

    # @转化成百分比
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)

    # @转化成length*length形状
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    # print(proportion)
    pshow = np.array(pshow).reshape(length, length)
    # print(pshow)

    # @画图：
    # @设置字体
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Times New Roman 字体路径
    font_prop = font_manager.FontProperties(fname=font_path)

    # @显示出矩阵
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.YlOrBr)
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, fontproperties=font_prop)
    plt.yticks(tick_marks, classes, fontsize=12, fontproperties=font_prop)

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                     weight=5, fontproperties=font_prop)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white', fontproperties=font_prop)
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, fontproperties=font_prop)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, fontproperties=font_prop)

    plt.ylabel('True label', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Predict label', fontsize=16, fontproperties=font_prop)
    plt.tight_layout()  # 自动调整间距
    plt.show()

if __name__ == '__main__':

    num_sub = 54
    num_epo = 600

    acc, std, sem, epo_best = acc_mean_std_multiple_sub_ave_tes(num_sub)
    all_sub_pdt = con_allPdt_with_best_acc_tes(epo_best)
    all_sub_tgt = con_allTgt_with_best_acc_tes(epo_best)
    ff = loss_tra_plot_multiple_sub_ave(num_sub, num_epo)

    all_sub_tgt_tmp = np.array(all_sub_tgt).flatten()  # 将N维度数组转成一维数组
    all_sub_pdt_tmp = np.array(all_sub_pdt).flatten()

    classes = ['class1', 'class2', 'class3', 'class4']  # 类别
    # CReport = classification_report_diy(all_sub_tgt_tmp, all_sub_pdt_tmp, classes)

    Cmatrix_tmp = getCMatrix(all_sub_tgt_tmp, all_sub_pdt_tmp)
    draw_pic_from_CMatrix(classes, Cmatrix_tmp)
    print("MATraditional Is OK!")
