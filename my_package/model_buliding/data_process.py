import pandas as pd
import numpy as np
import math as mt
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

# 读取数据
sDGP = np.loadtxt('../source/data/1126335/outside_0920/sDGP.txt')
sUDI = np.loadtxt('../source/data/1126335/outside_0920/sUDI.txt')
sd_angle = np.loadtxt('../source/data/1126335/angle_255.txt')
sd_angle = [round(mt.radians(angle), 2) for angle in sd_angle]
sd_position = np.loadtxt('../source/data/1126335/position_255.txt')
azimuth = df.iloc[:, 0]
altitude = df.iloc[:, 1]

copied_azimuth = []
copied_altitude = []

# 复制交替的列数据 255 次(循环复制)
for i in range(255):
    for item in azimuth:
        copied_azimuth.append(item)
    for item in altitude:
        copied_altitude.append(item)

# 重复复制遮阳数据4417次（单次复制）
copied_angle = np.repeat(sd_angle, 4417)
copied_interval = np.repeat(sd_position, 4417)

# print(copied_interval)
# print(copied_angle)
# print(len(copied_azimuth))
# print(len(copied_altitude))

# 生成连续的序号
index1 = pd.RangeIndex(start=1, stop=len(copied_azimuth) + 1)
index2 = pd.RangeIndex(start=0, stop=len(copied_azimuth))

# 创建新的 DataFrame，使用连续的序号作为索引
new_df = pd.DataFrame({'Azimuth': copied_azimuth})
new_df['Altitude'] = copied_altitude
new_df['Shade Angle'] = copied_angle
new_df['Shade Interval'] = copied_interval
new_df['sDGP'] = sDGP
new_df['sUDI'] = sUDI / 100

# 重新索引，将连续的序号赋给 DataFrame
new_df.index = index1
new_df.to_csv('../source/data/1126335/240920.csv', index=False)

normalized_data = normalize_data(new_df)
normalized_data.to_csv('../source/data/1126335/240920_normalized.csv')
# print(new_df)
# print(normalized_df)
