import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from load_dataset import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正确显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示正负号


def plot_loss(path: str, his_loss, his_val_loss, his_test_loss=None,
              Xlabel="Epochs", Ylabel="Loss", show=False):
    x = np.arange(0, len(his_loss))
    x = list(x)
    '''plot data'''
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率

    ax.plot(x, his_loss, "--", color='dodgerblue', marker='o', mec='r', mfc='w', label='Loss')
    ax.plot(x, his_val_loss, ":", color='peru', marker='^', mec='lightcoral', mfc='w', label='Val_loss')
    if his_test_loss != None:
        ax.plot(x, his_test_loss, "-.", color='purple', marker='s', mec='y', mfc='w', label='Test_loss')

    ax.set_xlabel(Xlabel, fontsize=20)
    ax.set_ylabel(Ylabel, fontsize=20)

    ax.tick_params(labelsize=15)  # 刻度字体大小

    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2), frameon=True, ncol=1)
    fig.set_tight_layout(tight='rect')
    plt.savefig(path)
    if show == True:
        plt.show()
    plt.close()


def plot_data(path: str, data1, label1=None,
              data2=None, label2=None,
              data3=None, label3=None,
              title="Title", Xlabel="Epochs", Ylabel="Loss", show=False):
    x = np.arange(0, len(data1))
    x = list(x)
    '''plot data'''
    fig, ax = plt.subplots(figsize=(10, 7))
    # plt.rcParams['savefig.dpi'] = 500  # 图片像素
    # plt.rcParams['figure.dpi'] = 500  # 分辨率

    # ax.plot(x, data1, "--", color='dodgerblue', marker='o', mec='r', mfc='w', label=label1)
    ax.plot(x, data1, "-", color='r', mfc='w', label=label1)
    if type(data2) != type(None):
        # ax.plot(x, data2, ":", color='peru', marker='^', mec='lightcoral', mfc='w', label=label2)
        ax.plot(x, data2, "-", color='lightcoral', mfc='w', label=label2)
    elif type(data3) != type(None):
        # ax.plot(x, data3, "-.", color='purple', marker='s', mec='y', mfc='w', label=label3)
        ax.plot(x, data3, "-", color='y', mfc='w', label=label3)

    # ax.set_title(title, fontsize=20)
    ax.set_xlabel(Xlabel, fontsize=20)
    ax.set_ylabel(Ylabel, fontsize=20)
    ax.tick_params(labelsize=15)  # 刻度字体大小

    fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.2), frameon=True, ncol=1)
    fig.set_tight_layout(tight='rect')
    plt.savefig(path)
    if show == True:
        plt.show()
    plt.close()


def plot_heatmap(cm, value=[0, 1]):
    sns.set_theme()
    # ax = sns.heatmap(cm, vmin=value[0], vmax=value[1])
    ax = sns.heatmap(cm, cmap="Blues", vmin=value[0], vmax=value[1])
    plt.show()
    # plt.close()



if __name__ == "__main__":

    print("end")

