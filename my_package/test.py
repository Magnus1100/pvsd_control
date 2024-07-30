import pandas as pd
import numpy as np

# 测试路径
# epw_data_file_path = './source/data/epw_data.csv'
# epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)
# print(epw_dataset)
#
# hoy = 8
# a = epw_dataset['Azimuth'][hoy]
# print(a)

# # 测试 hoylist 输出
# import shade_pygmo as sp
# start_date = "12-21"
# end_date = "12-21"
# skip_weekdays = False
# start_hour = 8
# end_hour = 17
# hoy_list = sp.hoyEditor.generateHoyList(start_date, end_date, skip_weekdays, start_hour, end_hour)  # 需要优化的HOY列表
# print(hoy_list)

# 测试欧式距离计算
# import shade_pygmo as sp
# pvsd_ED = sp.calculateED.GetPvsdED(0, 0, 45, 0.06, 16)
# print(pvsd_ED)

# # 测试占位符
# print("Hello %i World%i" % (200, 300))
# print('hahaha%.2f'%100.249879)

# 测试min，max
sDGP = np.loadtxt('source/data/sDGP.txt')
sUDI = np.loadtxt('source/data/sUDI.txt')
vis_data = pd.read_csv('source/dataset/vis_data.csv')
vis = vis_data['vis']
print('min_sdgp:%.2f' % min(sDGP))
print('max_sdgp:%.2f' % max(sDGP))
print('min_vis:%.2f' % min(vis))
print('max_vis:%.2f' % max(vis))
