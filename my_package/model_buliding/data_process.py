import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 标准化数据
def normalize_data(data):
    # 提取列名
    column_names = data.columns

    # 归一化数据
    scaler = MinMaxScaler()
    normalized_dataset = scaler.fit_transform(data)

    # 将归一化后的数据转换为 DataFrame，并将列名添加回来
    normalized_df = pd.DataFrame(normalized_dataset, columns=column_names)

    return normalized_df


# 读取数据文件
df = pd.read_csv('../source/data/dataset_found.csv', header=None)
df2 = pd.read_csv('../source/data/dataset_found (2).csv', header=None)
sDGP = np.loadtxt('../source/data/sDGP.txt')
sUDI = np.loadtxt('../source/data/sUDI.txt')

azimuth = df.iloc[:, 0]
altitude = df.iloc[:, 1]
shade_angle = df2.iloc[:, 0]
shade_interval = df2.iloc[:, 1]

copied_azimuth = []
copied_altitude = []

# 复制交替的列数据 349 次(循环复制)
for i in range(349):
    for item in azimuth:
        copied_azimuth.append(item)
    for item in altitude:
        copied_altitude.append(item)

# 重复复制遮阳数据4417次（单次复制）
copied_angle = np.repeat(shade_angle, 4417)
copied_interval = np.repeat(shade_interval, 4417)

# 生成连续的序号
index1 = pd.RangeIndex(start=1, stop=len(copied_azimuth) + 1)
index2 = pd.RangeIndex(start=0, stop=len(copied_azimuth))
copied_interval.index = index2
copied_angle.index = index2

# 创建新的 DataFrame，使用连续的序号作为索引
new_df = pd.DataFrame({'Azimuth': copied_azimuth})
new_df['Altitude'] = copied_altitude
new_df['Shade Angle'] = copied_angle
new_df['Shade Interval'] = copied_interval
new_df['sDGP'] = sDGP / 100
new_df['sUDI'] = sUDI / 100

# 重新索引，将连续的序号赋给 DataFrame
new_df.index = index1

# 归一化数据
normalized_data = normalize_data(new_df)
