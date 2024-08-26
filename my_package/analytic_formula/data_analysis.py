import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from collections import Counter

# 全局变量-数据
sDGP_1541553 = np.loadtxt('../source/data/sDGP.txt')
sDGP_1126335 = np.loadtxt('../source/data/1126335/240821_sDGP.txt')
sUDI = np.loadtxt('../source/data/sUDI.txt')
Vis = np.loadtxt('../source/data/vis_dataset_2821.txt')


def main():
    # 创建图形和子图
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    #
    # # sDGP 的 Seaborn 直方图和核密度估计图，密度
    # sns.histplot(sDGP, bins=30, stat='density', ax=axs[0, 0])
    # axs[0, 0].set_title('sDGP (Density)')
    # axs[0, 0].set_xlabel('Value')
    # axs[0, 0].set_ylabel('Density')
    #
    # # sDGP 的 Seaborn 直方图和核密度估计图，数量
    # sns.histplot(sDGP, bins=30, stat='count', ax=axs[0, 1])
    # axs[0, 1].set_title('sDGP (Count)')
    # axs[0, 1].set_xlabel('Value')
    # axs[0, 1].set_ylabel('Count')
    #
    # # sUDI 的 Seaborn 直方图和核密度估计图，密度
    # sns.histplot(sUDI, bins=30, stat='density', ax=axs[1, 0])
    # axs[1, 0].set_title('sUDI (Density)')
    # axs[1, 0].set_xlabel('Value')
    # axs[1, 0].set_ylabel('Density')
    #
    # # sUDI 的 Seaborn 直方图和核密度估计图，数量
    # sns.histplot(sUDI, bins=30, stat='count', ax=axs[1, 1])
    # axs[1, 1].set_title('sUDI (Count)')
    # axs[1, 1].set_xlabel('Value')
    # axs[1, 1].set_ylabel('Count')
    # a = np.mean(sDGP)
    # b = np.mean(sUDI)
    # c = np.mean(Vis)
    # print(a, b, c)
    #
    # # vis 的 Seaborn 直方图和核密度估计图，数量
    # # 创建图形
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # # 绘制直方图和核密度估计图，密度
    # sns.histplot(Vis, bins=30, stat='count', ax=ax)
    # ax.set_title('Vis (Density)')
    # ax.set_xlabel('Value')
    # ax.set_ylabel('Count')
    #
    # # 调整布局以避免重叠
    # plt.tight_layout()
    # # 显示图形
    # plt.show()

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图上绘制 sDGP_1541553 的分布
    ax1.hist(sDGP_1541553, bins=50, color='blue', edgecolor='black')
    ax1.set_title('Distribution of sDGP_1541553')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')

    # 在第二个子图上绘制 sDGP_1126335 的分布
    ax2.hist(sDGP_1126335, bins=50, color='red', edgecolor='black')
    ax2.set_title('Distribution of sDGP_1126335')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()

    # 每4417行转化成一列
    chunk_size = 4417
    num_chunks = len(sDGP_1126335) // chunk_size + int(len(sDGP_1541553) % chunk_size != 0)

    # 将数组 reshape 并转置
    reshaped_array = sDGP_1126335.reshape(num_chunks, chunk_size).T

    # 输出 reshaped_array 或 flattened_column
    print(reshaped_array.shape)
    print(reshaped_array)


if __name__ == '__main__':
    main()
