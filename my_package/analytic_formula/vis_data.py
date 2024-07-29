import numpy as np
import pandas as pd
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（假设 my_package 是项目根目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 构建 vis_dataset_2821.txt 的路径
vis_dataset_path = os.path.join(project_root, 'source', 'data', 'vis_dataset_2821.txt')

# 读取 vis_dataset_2821.txt 文件内容
with open(vis_dataset_path, 'r') as file:
    vis_data = file.readlines()

# 去除每行末尾的换行符
vis_data = [line.strip() for line in vis_data]

# 将文件内容传递给 DataFrame
vis_dataset = pd.DataFrame({'vis': vis_data})

sd_interval = np.linspace(-0.15, 0.15, 31)
sd_angle = np.linspace(0, 90, 91)

copied_angle = []
copied_interval = []

# 循环复制
for i in range(31):
    for item in sd_angle:
        copied_angle.append(item)

# 单次复制
copied_interval = np.repeat(sd_interval, 91)

vis_dataset['sd_angle'] = copied_angle
vis_dataset['sd_location'] = copied_interval.round(2)

# print(vis_dataset)
