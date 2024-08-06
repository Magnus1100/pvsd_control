import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# 全局变量-数据
sDGP = np.loadtxt('../source/data/sDGP.txt')
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
    a = np.mean(sDGP)
    b = np.mean(sUDI)
    c = np.mean(Vis)
    print(a, b, c)

    # vis 的 Seaborn 直方图和核密度估计图，数量
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制直方图和核密度估计图，密度
    sns.histplot(Vis, bins=30, stat='count', ax=ax)
    ax.set_title('Vis (Density)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')

    # 调整布局以避免重叠
    plt.tight_layout()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    main()
