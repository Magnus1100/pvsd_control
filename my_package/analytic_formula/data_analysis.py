import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from collections import Counter
from scipy import stats

# 全局变量-数据
sDGP_1126335 = np.loadtxt('../source/data/1126335/outside_0920/sDGP.txt')
sUDI_1126335 = np.loadtxt('../source/data/1126335/outside_0920/sUDI.txt')
Vis = np.loadtxt('../source/data/vis_dataset_2821.txt')
sUDI_1126335 = sUDI_1126335/100

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
    # 计算平均数（均值）
    # mean_sDGP = np.mean(sDGP_1126335)
    # mean_sUDI = np.mean(sUDI_1126335)
    # # 计算中位数
    # median_sDGP = np.median(sDGP_1126335)
    # median_sUDI = np.median(sUDI_1126335)
    # # 计算标准差
    # std_dev_sDGP = np.std(sDGP_1126335)
    # std_dev_sUDI = np.std(sUDI_1126335)
    # # 计算方差
    # variance = np.var(sUDI_1126335)
    # # 计算四分位数（25th, 50th, 75th）
    # q25, q50, q75 = np.percentile(sUDI_1126335, [25, 50, 75])
    # # 计算偏度（反映分布的对称性）
    # skewness = stats.skew(sUDI_1126335)
    # # 计算峰度（反映分布的陡峭程度）
    # kurtosis = stats.kurtosis(sUDI_1126335)
    #
    # # 输出结果
    # print(f"平均数 (Mean): {mean_sUDI}")
    # print(f"中位数 (Median): {median_sUDI,}")
    # print(f"标准差 (Standard Deviation): {std_dev_sUDI}")
    # print(f"方差 (Variance): {variance}")
    # print(f"25th 百分位 (Q1): {q25}")
    # print(f"50th 百分位 (Median): {q50}")
    # print(f"75th 百分位 (Q3): {q75}")
    # print(f"偏度 (Skewness): {skewness}")
    # print(f"峰度 (Kurtosis): {kurtosis}")

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图上绘制 sDGP_1541553 的分布
    ax1.hist(sDGP_1126335, bins=50, color='black', edgecolor='black')
    ax1.set_title('sDGP数据分布')
    ax1.set_xlabel('序号')
    ax1.set_ylabel('sDGP')

    # 在第二个子图上绘制 sDGP_1126335 的分布
    ax2.hist(sUDI_1126335, bins=75, color='black', edgecolor='black')
    ax2.set_title('sUDI数据分布')
    ax2.set_xlabel('序号')
    ax2.set_ylabel('sUDI')

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
