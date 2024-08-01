import pandas as pd
import numpy as np
import polars as pl
import shade_pygmo as sp
from scipy.spatial import distance

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
# sDGP = np.loadtxt('source/data/sDGP.txt')
# sUDI = np.loadtxt('source/data/sUDI.txt')
# vis_data = pd.read_csv('source/dataset/vis_data.csv')
# vis = vis_data['vis']
# print('min_sdgp:%.2f' % min(sDGP))
# print('max_sdgp:%.2f' % max(sDGP))
# print('min_vis:%.2f' % min(vis))
# print('max_vis:%.2f' % max(vis))

# 测试data_collector
# # a = range(11)
# # 生成 1 到 10 的数组
# a = np.arange(1, 11)
# # 重复数组 10 次，形成一个长度为 100 的数组
# repeated_index = np.tile(a, 10)
# # 创建一个空的 DataFrame
# df = pd.DataFrame({'generate:': repeated_index})
# # 设置 DataFrame 的索引
# df.index = repeated_index
# print(df)
# df.to_csv('test.csv')

# data_collector = pd.DataFrame(
#     columns=['sdgp:', 'sdgp_valued:', 'sUDI:', 'sUDI_valued:', 'vis', 'vis_valued:', 'pvg', 'pvg_valued:', 'ED',
#              'ED_valued:'])
# a = 0.777
# new_row = pd.DataFrame({
#     'sdgp:': [0.7],
#     'sdgp_valued:': [0.1],
#     'sUDI:': [0.1],
#     'sUDI_valued:': [0.1],
#     'vis': [0.1],
#     'vis_valued:': [0.1],
#     'pvg': [0.1],
#     'pvg_valued:': [0.1],
#     'ED': [0.1],
#     'ED_valued:': [0.1]
# })
#
# data_collector = pd.concat([data_collector, new_row], ignore_index=True)
# print(data_collector)
# 示例数据
# schedule = {
#     1: {'sd_angle': 30, 'sd_site': 'SiteA', 'best_fitness': 0.95},
#     2: {'sd_angle': 45, 'sd_site': 'SiteB', 'best_fitness': 0.92},
#     3: {'sd_angle': 60, 'sd_site': 'SiteC', 'best_fitness': 0.85},
# }
# # 提取 HOY 列
# hoy_values = list(schedule.keys())
# # 创建 DataFrame
# schedule_df = pl.DataFrame({'HOY': hoy_values})
# schedule_df = schedule_df.with_columns([
#     pl.Series('sd_angle', [entry['sd_angle'] for entry in schedule.values()]),
#     pl.Series('sd_site', [entry['sd_site'] for entry in schedule.values()]),
#     pl.Series('best_fitness', [entry['best_fitness'] for entry in schedule.values()])
# ])
#
# print("Schedule DataFrame:")
# print(schedule_df)

# 测试hoy结合
# spring_date, summer_date, autumn_date, winter_date = "3-21", "6-21", "9-21", "12-21"  # 典型日期
#
# springDay_hoy = sp.hoyEditor.generateHoyList(spring_date, spring_date, False)  # 春分
# summerDay_hoy = sp.hoyEditor.generateHoyList(summer_date, summer_date, False)  # 夏至
# autumnDay_hoy = sp.hoyEditor.generateHoyList(autumn_date, autumn_date, False)  # 秋分
# winterDay_hoy = sp.hoyEditor.generateHoyList(winter_date, winter_date, False)  # 冬至
# print(springDay_hoy, summerDay_hoy, autumnDay_hoy, winterDay_hoy)
#
# main_hoy = springDay_hoy + summerDay_hoy + autumnDay_hoy + winterDay_hoy  # 需要优化的HOY列表
# print(main_hoy)

# 测试欧式距离
# a = sp.calculateED.GetAxis(0, 0.12, 0, 16)
# print(a)
# b = sp.calculateED.GetED(0, 1, 0, 2)
# print(b)
# c = distance.euclidean((0, 0, 0), (0, 1, 2))
# print(c)

# df = pd.DataFrame(columns=['a', 'b', 'c'])
# a = pd.DataFrame({
#     'b': [2],
#     'c': [3]
# })
# new_df = pd.concat([df, a], ignore_index=True)
# new_df2 = pd.concat([df, a], ignore_index=True)
# new_df3 = pd.concat([df, a], ignore_index=True)
# new_df4 = pd.concat([df, a], ignore_index=True)
# print(new_df4)

df = pd.DataFrame(
    columns=['sdgp:', 'sdgp_valued:', 'sUDI:', 'sUDI_valued:', 'vis', 'vis_valued:', 'pvg', 'pvg_valued:', 'ED',
             'ED_valued:'])

step_data = pd.DataFrame({
    'sdgp': [1],
    'sUDI': [1],
    'vis': [1],
    'pvg': [1],
    'ED': [1],
    'sdgp_valued': [1],
    'sUDI_valued': [1],
    'vis_valued': [1],
    'pvg_valued': [1],
    'ED_valued': [1]
})

df = pd.concat([df,step_data], ignore_index=True)
df.to_csv('test.csv')
print(df)
