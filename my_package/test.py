import pandas as pd
import numpy as np
import polars as pl
# import shade_optimizer as sp
from scipy.spatial import distance
import pygmo as pg

# 测试路径
# epw_data_file_path = './source/data/epwData_sz.csv'
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
# vis_data = pd.read_csv('source/dataset/vis_data_outside_0920.csv')
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

# df = pd.DataFrame(
#     columns=['sdgp:', 'sdgp_valued:', 'sUDI:', 'sUDI_valued:', 'vis', 'vis_valued:', 'pvg', 'pvg_valued:', 'ED',
#              'ED_valued:'])
#
# step_data = pd.DataFrame({
#     'sdgp': [1],
#     'sUDI': [1],
#     'vis': [1],
#     'pvg': [1],
#     'ED': [1],
#     'sdgp_valued': [1],
#     'sUDI_valued': [1],
#     'vis_valued': [1],
#     'pvg_valued': [1],
#     'ED_valued': [1]
# })
#
# df = pd.concat([df,step_data], ignore_index=True)
# df.to_csv('test.csv')
# print(df)
#
# import pandas as pd
#
# # 创建两个示例 DataFrame
# df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
#
# # 使用 assign 方法
# df1 = df1.assign(C=df2['C'], D=df2['D'])
#
# print(df1)
# a = pg.problem

# a = sp.DataCollector()
#
# # 添加数据条目
# sp.data_collector.add_entry({
#     'generation': 1,
#     'individual_index': 1,
#     'sd_angle': 30,
#     'sd_location': 'location1',
#     'sDGP': 0.5,
#     'sUDI': 0.7,
#     'vis': 0.9,
#     'pvg': 0.3,
#     'ED': 0.6,
#     'sDGP_valued': 0.8,
#     'sUDI_valued': 0.9,
#     'vis_valued': 1.0,
#     'pvg_valued': 0.4,
#     'ED_valued': 0.7
# })

# class DataCollector:
#     def __init__(self):
#         self.columns = ['generation', 'individual_index', 'sd_angle', 'sd_location',
#                         'sDGP', 'sUDI', 'vis', 'pvg', 'ED',
#                         'sDGP_valued', 'sUDI_valued', 'vis_valued', 'pvg_valued', 'ED_valued']
#         self.data = []
#
#     def add_entry(self, entry):
#         self.data.append(entry)
#
#     def to_dataframe(self):
#         return pd.DataFrame(self.data, columns=self.columns)
#
#     def to_csv(self, filename):
#         df = self.to_dataframe()
#         df.to_csv(filename, index=False)
#         print(f"Data exported to {filename}")
#
# # 使用示例
# # data_collector = DataCollector()
#
# # 添加数据条目
# data_collector.add_entry([
#     1, 1, 30, 'location1', 0.5, 0.7, 0.9, 0.3, 0.6, 0.8, 0.9, 1.0, 0.4, 0.7
# ])
# data_collector.add_entry([
#     1, 2, 45, 'location2', 0.6, 0.8, 0.95, 0.35, 0.65, 0.85, 0.95, 1.05, 0.45, 0.75
# ])
#
# print(data_collector)
# # 导出到 Excel 文件
# data_collector.to_csv('output.csv')

# pop_size = 3
# gen_size = 4
# gen_list = np.repeat(range(gen_size + 1), pop_size)
# pop_list = []
# for i in range(gen_size + 1):
#     for item in range(pop_size):
#         pop_list.append(item)
# a = pd.DataFrame()
# a['gen'] = gen_list
# a['pop'] = pop_list
# print(a)
#
# sp.save_dataframe(a, 'a')


# import pygmo as pg
#
#
# class MyProblem:
#     def __init__(self):
#         # 定义问题的维度，例如这里假设问题是2维的
#         self.dim = 2
#
#     # pygmo要求定义以下方法
#     def fitness(self, x):
#         # 定义目标函数，例如我们定义一个简单的二次函数作为目标
#         return [x[0] ** 2 + x[1] ** 2]
#
#     def get_bounds(self):
#         # 定义决策变量的上下界，例如我们定义为[-5, 5]区间
#         return ([-5] * self.dim, [5] * self.dim)
#
#     def get_name(self):
#         return "My Custom Problem"
#
#     def get_nobj(self):
#         return 1  # 这里我们定义一个单目标优化问题
#
#     def get_nix(self):
#         return self.dim  # 决策变量的维度
#
#
# # 假设你已经定义好了问题和算法
# prob = pg.problem(MyProblem())  # 替换为你的问题
# algo = pg.algorithm(pg.sade(gen=100))  # 选择你的算法并设置生成代数
#
# # 创建进化岛
# isl = pg.island(algo=algo, prob=prob, size=20)
#
# # 执行演化
# isl.evolve()
# isl.wait_check()
#
# # 获取种群
# pop = isl.get_population()
#
# # 获取适应度最高的个体
# best_idx = pop.best_idx()
# best_individual = pop.get_x()[best_idx]
# best_fitness = pop.get_f()[best_idx]
#
# print("适应度最高的个体:", best_individual)
# print("该个体的适应度值:", best_fitness)
# fitness = [0.5, 0.6]
# a = [
#     {
#         'id': [1, 2],
#         'fitness': 0.5
#     },
#     {
#         'id': [3, 4],
#         'fitness': 0.6
#     }
# ]
#
# min_fitness = min(fitness)
# best_individuals = [ind for ind in a if ind['fitness'] == min_fitness]
# print(best_individuals)
#
# best_form = best_individuals[0]['id']
# best_angle = best_form[0]
# best_loc = best_form[1]
# print(best_form, best_angle, best_loc)

# 获取最优形变
# index = [1, 2, 3]
# individual_best_angle = [45, 90, 135]
# individual_best_location = [-0.1, -0.2, -0.3]
# ED = [0.1, 0.2, 0.3]
#
# best_individual_list = {
#         'index': index,
#         'angle': individual_best_angle,
#         'location': individual_best_location,
#         'ED': ED
#     }
# print(best_individual_list)
# list_fit = pd.DataFrame(best_individual_list)
# # 定义要匹配的ED值
# best_ED = 0.2
#
# # 筛选ED等于best_ED的行，并提取对应的angle和location
# selected_rows = list_fit[list_fit['ED'] == best_ED]
# angles = selected_rows['angle'].values
# locations = selected_rows['location'].values
#
# # 打印结果
# print("Selected Angles:", angles)
# print("Selected Locations:", locations)

# a = [2, 2, 3, 1, 4, 5, 1]
# B = [1, 2, 3, 4, 5, 6, [1,7]]
#
# C = pd.DataFrame({
#     'a': a,
#     'B': B
# })
#
# D = min(a)
# # 筛选出 a 列等于 D 的所有行，并提取 B 列的值
# E = C[C['a'] == D]['B'].values
# print(E)

# import pickle
#
# # 尝试加载已有的数据，如果文件不存在则初始化一个空列表
# try:
#     with open('saved_values.pkl', 'rb') as f:
#         values = pickle.load(f)
# except FileNotFoundError:
#     values = []
#
# # 添加新的值（假设是本次运行生成的值）
# new_value = 54  # 示例值，可以是任何计算或输出的结果
# values.append(new_value)
#
# # 将更新后的数据保存回文件
# with open('saved_values.pkl', 'wb') as f:
#     pickle.dump(values, f)
#
# print("Updated values:", values)

# import pickle
#
# # 尝试加载已有的数据，如果文件不存在则提示未找到旧值
# try:
#     with open('saved_values.pkl', 'rb') as f:
#         old_values = pickle.load(f)
#         print("Old values:", old_values)
# except FileNotFoundError:
#     old_values = None
#     print("No previous values found.")
#
# # 生成新值，这个值会替代之前保存的值
# new_values = [0.12,0.22]  # 示例值，可以是本次运行生成的任何数据
#
# # 将新的数据保存到文件中，覆盖之前的数据
# with open('saved_values.pkl', 'wb') as f:
#     pickle.dump(new_values, f)
#
# print("Saved new values:", new_values)

# a = range(8, 17)
# for a in a:
#     print(a)
# a = range(9,17)
# df = pd.DataFrame(columns=['Hoy', 'Angle', 'Position'])
# # 显示空的 DataFrame
# for i in a:
#     df.loc[a.index(i)] = [1, 45.0, 0.12]  # 添加第一行
#
# print(df)
# import math
#
#
# class CalculateBlindAxis:
#     @staticmethod
#     def CalculatePoints(p1, p2, shade_length, shade_angle, shade_position, blind_count):
#         shade_angle = math.radians(shade_angle)
#         shade_interval = abs(shade_position)
#         # print("shade_interval:"+str(shade_interval))
#         y_changed = shade_length * math.sin(shade_angle)
#         z_changed = shade_length * math.cos(shade_angle)
#         print(y_changed, z_changed)
#
#         if shade_position <= 0:
#             point1 = [p1[0], p1[1], p1[2]]
#             point2 = [p1[0], p1[1] + y_changed, p1[2] + z_changed]
#             point3 = [p2[0], p1[1] + y_changed, p2[2] + z_changed]
#             point4 = [p2[0], p1[1], p1[2]]
#         else:
#             print(p1[2],shade_interval * (blind_count - 1))
#             point1 = [p1[0], p1[1], p1[2] - shade_interval * (blind_count - 1)]
#             point2 = [p1[0], p1[1] + y_changed, p1[2] + z_changed - shade_interval * (blind_count - 1)]
#             point3 = [p2[0], p1[1] + y_changed, p2[2] + z_changed - shade_interval * (blind_count - 1)]
#             point4 = [p2[0], p1[1], p1[2] - shade_interval * (blind_count - 1)]
#
#         point_list = [point1, point2, point3, point4]
#         return point_list
#
#     @staticmethod
#     def GetBlindAxis(point_list, blind_count, shade_position):
#         blind_axis = []
#         shade_interval = 0.15 - abs(shade_position)
#         print(shade_interval)
#
#         for i in range(blind_count):
#             adjusted_points = []
#             for point in point_list:
#                 new_point = [point[0], point[1], point[2] - shade_interval * i]
#                 adjusted_points.append(new_point)
#             blind_axis.append(adjusted_points)
#         return blind_axis
#
#
# fp = [[32.030005, -43.670005, 2.83],
#       [29.990005, -43.61625, 2.970037]]

# po1, po2 = fp[0], fp[1]
# bc, sl = 16, 0.15
# sp, sa = 0.02, 65
#
# top_pl = CalculateBlindAxis.CalculatePoints(po1, po2, sl, sa, sp, bc)
# print(top_pl)
# bpl_test = CalculateBlindAxis.GetBlindAxis(top_pl, bc, sp)
#
# # 方法 2: 使用 pprint 模块格式化输出
# import pprint
#
# pprint.pprint(bpl_test)
#

# import math
# print(math.cos(1.48353))
#
# print(math.cos(85))

# data = "{-0.831203,0.55312,-0.056211}"
# # 替换大括号为中括号
# data_list = data.replace("{", "[").replace("}", "]")
# # 转换为 Python 的列表对象
# result = eval(data_list)
#
# print(result)  # 输出：[-0.831203, 0.55312, -0.056211]

import math
shading_angles = range(0,90)
for shading_angle in shading_angles:

    surface_normal = [0, math.cos(math.radians(180 - shading_angle)), math.sin(math.radians(shading_angle))]
    print(surface_normal)
