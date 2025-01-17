import os
import time
import joblib
import math as mt
import numpy as np
import pandas as pd
import pygmo as pg
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime_gen import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 自己的一些算法，主要计算板之间的遮挡和光伏发电
import blind_shade_calculate as bsc
import pvg_calculate as pc

"""
功能：通过调用机器学习模型和优化算法，生成某一段时间的遮阳形态参数
使用步骤：
1.设置各文件路径
2.设置模拟时间HOY
3.设置模拟权重：眩光|采光|视野|发电
4.点击运行输出最优形态 -> 做能耗和采光分析

代码结构：
1.类dataSaver-把dataframe输出成csv：方法get_unique_filename-生成文件名；方法save_dataframe
2.类hoyEditor-生成一个hoy列表（输入开始/结束日期）生成一段时间的hoy列表
"""

# >>> 声明实例 <<<
aim_location = 'sz'  # 目标地 bj|sz|km|hb
aim_hoy = 'annual_hoy'  # 目标时间 hoy1-12|annual_hoy
current_datetime = datetime.now().strftime('%Y%m%d_%H%M')  # 获取当前日期和时间，精确到分钟
used_model = f'model_sz-250107'  # 使用的模型，在model_optimizer中

# >>> 输出路径 <<<
base_schedule_name = f'{aim_location}_1100_schedule_{aim_hoy}'
schedule_name = f"{base_schedule_name}_{current_datetime}"
schedule_output_path = f'./shading_schedule/sz-1100-250117/{schedule_name}.csv'
log_output_path = f'./shading_schedule/sz-1100-250117/log'  # 日志输出路径

# >>> pvsd实例 <<<
pvsd_instance = bsc.pvShadeBlind(0.15, 2.1, 4.896, 100, 0
                                 , 16, 2.4)
# >>> 读取数据 <<<
vis_data = pd.read_csv(r'./source/data/data_shadeCalculate/vis_data_outside_0920.csv')
epw_data_file_path = fr'./source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}_withPVG.csv'
epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)
print(f'so - Have read epw data:{epw_data_file_path}')

Azimuth = epw_dataset['Azimuth']
Altitude = epw_dataset['Altitude']
Radiation = epw_dataset['Direct_Rad']

sDGP = np.loadtxt(fr'./source/data/data_mlTrain/{aim_location}/{aim_location}_sDGP_nearWindow.txt')
sUDI = np.loadtxt(fr'./source/data/data_mlTrain/{aim_location}/{aim_location}_sUDI.txt')

# >>> 读取ml模型 <<<
model_sdgp = joblib.load(f'./source/model_optimizer/{used_model}/sDGP_RF.pkl')
model_sudi = joblib.load(f'./source/model_optimizer/{used_model}/sUDI_RF.pkl')
print(f'so - Have read ml model:{used_model}')

# >>> 全局变量 <<<
all_data = []  # 数据记录
optimizer_data = []  # 优化数据记录
shade_schedule = pd.DataFrame(columns=['Hoy', 'SD_Angle', 'SD_Position'])

# >>> ！！！重要变量！！！ <<<
main_hoy_path = f'source/data/hoys/hoy_{aim_location}/{aim_hoy}.txt'
main_hoy = np.loadtxt(main_hoy_path)
print(f'so - Have read hoy data:{main_hoy_path}')

weight_dgp, weight_udi, weight_vis, weight_pvg = 1, 1, 0, 0  # 各项权重[0,1]
pygmo_gen, pygmo_pop = 10, 10  # 迭代次数，每代人口

# 取值范围
min_angle, max_angle = mt.radians(0), mt.radians(90)  # 角度范围
min_position, max_position = -0.14, 0.14  # 位置范围
min_azimuth, max_azimuth = mt.radians(min(Azimuth)), mt.radians(max(Azimuth))  # 方位角范围
min_altitude, max_altitude = mt.radians(min(Altitude)), mt.radians(max(Altitude))  # 高度角范围
min_radiation, max_radiation = min(Radiation), max(Radiation)  # 👈新加的特征值

print('angle_round:', [min_angle, round(max_angle, 3)])
print('azimuth_round:', [round(min_azimuth, 3), round(max_azimuth, 3)])
print('altitude_round:', [round(min_altitude, 3), round(max_altitude, 3)])
print('radiation_round:', [round(min_radiation, 3), round(max_radiation, 3)])


class dataSaver:  # 获取唯一的文件名字。应用在最后的csv生成中。
    @staticmethod
    def get_unique_filename(base_name, extension, counter):
        today_date = datetime.now().strftime('%Y%m%d')  # 当前日期格式为YYYYMMD
        return f'{base_name}_{counter}.{extension}'

    # 将dataframe保存在可以自己命名的csv文件里。如果名字重复则生成一个新文件。
    @staticmethod
    def save_dataframe(df, base_name, extension='csv', start_counter=1):
        counter = start_counter
        while True:
            filename = dataSaver.get_unique_filename(base_name, extension, counter)
            if not os.path.exists(filename):
                try:
                    df.to_csv(f'{log_output_path}/{aim_location}_{aim_hoy}_{current_datetime}.csv', index=False)
                    print(f'文件已保存为 {log_output_path}')
                    break
                except FileExistsError:
                    counter += 1  # 如果文件已存在，则增加计数器
            else:
                counter += 1  # 如果文件已存在，则增加计数器


def visualizeFitness(results, hoy_list):
    """
    展示不同HOY的平均适应度值随代数变化的图表。

    参数:
    - results (list): 包含每个HOY优化结果的元组列表。
    - hoy_list (list): 被优化的HOY列表。
    """
    cmap = plt.get_cmap('viridis')  # 获取viridis色图
    colors = cmap(np.linspace(0.2, 0.8, len(hoy_list)))  # 生成同一色系的不同深浅颜色

    legend_labels = []  # 用于存储图例标签
    hoy_values = []  # 用于存储对应的 hoy 值

    for i, (hoy, _, _, _, all_fitness) in enumerate(results):
        avg_fitness_per_generation = []

        # 计算每代的平均 fitness
        for gen_fitness in all_fitness:
            avg_fitness = np.mean(gen_fitness)
            avg_fitness_per_generation.append(avg_fitness)

        # 绘制每代的折线图
        generations = np.arange(len(avg_fitness_per_generation))
        plt.plot(generations, avg_fitness_per_generation, marker='o', markersize=4, color=colors[i])  # 调整标点大小

        # 收集图例标签和对应的 hoy 值
        legend_labels.append(f"Hoy {hoy}")
        hoy_values.append(hoy)

    # 根据 hoy 值的顺序重新排列图例
    sorted_indices = np.argsort(hoy_values)
    sorted_labels = [legend_labels[i] for i in sorted_indices]

    plt.xlabel('Generation')
    plt.ylabel('A+verage Fitness')
    plt.title('Average Fitness Values Over Generations for Different HOYs')
    plt.legend(sorted_labels)  # 使用排序后的图例

    plt.grid()  # 添加网格线
    plt.show()


def UpdateGenAndInd(data_list, gen_size, pop_size, hoy_list):
    # 生成 Hoy、Gen 和 Pop 列表
    hoy_list_extended = np.repeat(hoy_list, (gen_size + 1) * pop_size)  # HOY 列表扩展以匹配 gen 和 pop
    gen_list = np.tile(np.repeat(range(gen_size + 1), pop_size), len(hoy_list))  # 每个 HOY 包含多个 gen，每个 gen 包含多个 pop
    pop_list = np.tile(list(range(pop_size)), len(hoy_list) * (gen_size + 1))  # 每个 gen 包含多个 pop

    # 将生成的列表添加到 DataFrame
    data_list['Hoy'] = hoy_list_extended
    data_list['Gen'] = gen_list
    data_list['Ind'] = pop_list

    return data_list


def normalizeValue(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value


class calculateED:
    @staticmethod  # 根据遮阳状态计算相对坐标
    def GetAxis(sd_angle, sd_location, sd_index, ED_slat_count):
        # 角度转弧度
        sd_angle = mt.radians(sd_angle)
        l_slat = pvsd_instance.sd_width
        hw = pvsd_instance.window_height

        # 计算坐标（相对坐标）
        x_axis = mt.sin(sd_angle)
        if sd_location >= 0:  # loc0.01 第3块板 0.04
            z_axis = hw - l_slat * (1 - mt.cos(sd_angle)) + sd_location * (sd_index + 1)
        else:  # loc-0.01 第15块板 -0.01
            z_axis = hw - l_slat * (1 - mt.cos(sd_angle)) + sd_location * (ED_slat_count - sd_index - 1)

        return x_axis, z_axis

    @staticmethod  # 根据坐标计算欧式距离
    def GetED(a_origin, a_next, b_origin, b_next):
        ED = mt.sqrt(mt.pow(a_next - a_origin, 2) + mt.pow(b_next - b_origin, 2))
        return ED

    @staticmethod  # 计算两个坐标的欧式距离
    def GetPvsdED(angle_origin, loc_origin, angle_next, loc_next):
        # 总欧氏距离
        total_ED = 0
        sd_count = pvsd_instance.slat_count
        # 每个版单独计算
        for i in range(0, sd_count):
            x_origin, z_origin = calculateED.GetAxis(angle_origin, loc_origin, i, sd_count)  # 原始坐标
            x_next, z_next = calculateED.GetAxis(angle_next, loc_next, i, sd_count)  # 下个状态坐标

            ED_i = calculateED.GetED(x_origin, x_next, z_origin, z_next)  # 计算总欧式距离
            total_ED += ED_i
        return total_ED


class MyProblem:
    def __init__(self, hoy, azimuth, altitude, ver_angle, hor_angle, my_weights, max_pvg, radiation):
        self.n_var = 2  # 两个变量：sd_interval, sd_angle
        self.n_obj = 1  # 单目标优化

        # 存储每一代最优个体
        self.previous_best_angle = None
        self.previous_best_loc = None

        self.hoy = hoy  # 当前时间
        self.Azimuth = azimuth  # 太阳方位角
        self.Altitude = altitude  # 太阳高度角
        self.ver_angle = ver_angle  # 表面和太阳向量的垂直夹角
        self.hor_angle = hor_angle  # 表面和太阳向量的水平夹角
        self.my_weights = my_weights  # 权重值
        self.max_pvg = max_pvg  # 不同时刻最大发电量（用以标准化发电量）
        self.radiation = radiation  # 不同时刻的直射太阳辐射
        self.fitness_history = []  # 保存每一步的适应度
        self.optimize_history = []

        self.data_collector = []  # 保存每一步的数据条目

    def fitness(self, x):
        # 生成值变量
        sd_angle, sd_location = x
        sd_location = sd_location.round(2)
        # 计算板间距
        sd_interval = (0.15 - abs(sd_location)).round(2)  # 间距
        sd_angle_degree = int(mt.degrees(sd_angle))

        # 调用外部的机器学习模型进行预测
        # 归一化数据
        normalized_azimuth = normalizeValue(self.Azimuth, min_azimuth, max_azimuth)
        normalized_altitude = normalizeValue(self.Altitude, min_altitude, max_altitude)
        normalized_angle = normalizeValue(sd_angle, min_angle, max_angle)
        normalized_position = normalizeValue(sd_location, min_position, max_position)
        normalized_radiation = normalizeValue(self.radiation, min_radiation, max_radiation)
        # 预测特征序列
        predict_parameter = [normalized_azimuth, normalized_altitude, normalized_angle, normalized_position,
                             normalized_radiation]
        feature_names = ['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval', 'Direct Radiation']
        predict_parameters = pd.DataFrame([predict_parameter], columns=feature_names)

        pvsd = pvsd_instance
        """
        计算各优化值
        1, sdgp(0-1) : 眩光概率>0.38的空间比例，ml预测
        2, sudi(0-1) : 日光舒适比例照度[200-800lx]空间比例，ml预测
        3, vis(0-1) : 室内平均视野水平，数据库调用
        4, pvg_normal(0-1) : 归一化的光伏发电量（计算值/最大值），公式计算
        5, ED_percent : 形变的欧式距离，用在一定程度上减少形变
        - weighted_vals : 加权得到的优化值
        """
        # 1，2 - sudi/sdgp 机器学习模型预测
        pred_sdgp = model_sdgp.predict(predict_parameters)[0]
        normalized_sdgp = normalizeValue(pred_sdgp, min(sDGP), max(sDGP))  # 标准化sDGP
        val_sdgp = normalized_sdgp * self.my_weights[0]

        pred_sudi = model_sudi.predict(predict_parameters)[0]
        val_sudi = pred_sudi * self.my_weights[1]

        # 3 - 数据库调用查询vis
        vis = bsc.ShadeCalculate.GetVis(sd_angle_degree, sd_location)
        vis = float(vis[0])
        val_vis = vis * self.my_weights[2]

        # 4 - 调用公式计算pv发电量
        shade_percent = bsc.ShadeCalculate.AllShadePercent(pvsd.sd_length, pvsd.sd_width, sd_interval, self.ver_angle,
                                                           self.hor_angle, sd_angle)
        shade_rad = pc.pvgCalculator.calculateRadiant(sd_angle, self.hoy)
        pvg_value = pc.pvgCalculator.calculateHoyPVGeneration(self.hoy, shade_rad, pvsd.panel_area,
                                                              pvsd.P_stc) * (1 - shade_percent)
        normalized_pvg = pvg_value / self.max_pvg
        val_pvg = normalized_pvg * self.my_weights[3]

        # final value - 加权优化值
        val_all = val_sdgp + val_sudi + val_vis + val_pvg
        val_optimize = - val_all
        # print(val_all)
        # print(val_sdgp, val_sudi, val_vis, val_pvg)

        # 保存每一步个体形态和适应度
        self.fitness_history.append(val_optimize)
        self.optimize_history.clear()
        self.optimize_history.append({
            'shade_form': x,
            'fitness': val_optimize
        })

        # 保存每一代每个个体数据
        round_size = 3
        self.data_collector.append({
            'Hoy': self.hoy,  # 基本信息 <<<
            'Gen': 0,
            'Ind': 0,
            'Sd_A': sd_angle_degree,
            'Sd_L': sd_location,
            'SDGP': round(pred_sdgp, round_size),  # sDGP <<<
            'SUDI': round(pred_sudi, round_size),  # sUDI <<<
            'Vis': round(vis, round_size),  # VIS <<<
            'Pvg': round(pvg_value, round_size),  # PVG <<<
            'Val_SDGP': round(val_sdgp, round_size),
            'Val_SUDI': round(val_sudi, round_size),
            'Val_Vis': round(val_vis, round_size),
            'Val_Pvg': round(val_pvg, round_size),
            'shade_percent': round(shade_percent, round_size),
            'shade_rad': round(shade_rad, round_size),
            'Optimizer': round(val_optimize, round_size)
        })
        all_data.append(self.data_collector.copy())
        self.data_collector.clear()

        # ========== 打印结果 ==========
        # print('sd_angle: ' + str(sd_angle_degree))
        # print('sd_location: ' + str(sd_location))
        # print('weighted_vals: %2f' % abs(val_optimize))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # ========== 打印结果 ==========
        return [val_optimize]

    @staticmethod
    def get_bounds():
        return [min_angle, min_position], [max_angle, max_position]

    @staticmethod
    def get_names():
        return ['sd_angle', 'sd_site']

    # 更新上一代最优个体
    def update_previous_best(self, angle, loc):
        self.previous_best_angle = angle
        self.previous_best_loc = loc


class shade_pygmo:
    @staticmethod
    def optimize_hoy(hoy, epw_dataset, my_weights, gen_size=pygmo_gen, pop_size=pygmo_pop):

        # 垂直角，水平角
        my_ver_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Ver_Angle')
        my_hor_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Hor_Angle')
        # 方位角
        my_azimuth = epw_dataset.loc[hoy, 'Azimuth']
        my_azimuth = mt.radians(my_azimuth)
        # 高度角
        my_altitude = epw_dataset.loc[hoy, 'Altitude']
        my_altitude = mt.radians(my_altitude)
        # 直射太阳辐射
        my_radiation = epw_dataset.loc[hoy, 'Direct_Rad']

        try:
            # 尝试获 取 max_pv_generation 值
            max_pv_value = epw_dataset.loc[hoy, 'Max_Radiant(W/m2)']
            my_max_pvg = max_pv_value if max_pv_value != 0 else 1  # 检查值是否为零
        except KeyError:
            print(f"Value for HOY: {hoy} not found in the dataset.")

        # 声明优化问题实例
        problem_instance = MyProblem(hoy, my_azimuth, my_altitude, my_ver_angle, my_hor_angle, my_weights, my_max_pvg,
                                     my_radiation)
        prob = pg.problem(problem_instance)

        # 粒子群优化算法
        algo = pg.algorithm(pg.pso(gen=1))
        pop = pg.population(prob, size=pop_size)

        # 创建种群
        all_fitness = []  # 用于保存所有代的适应度值
        all_solution = []  # 保存所有代个体参数

        # 👇=========== 进行优化  =============👇
        for gen in range(gen_size):
            # 进化一代
            pop = algo.evolve(pop)

            # 获取当前代适应度值和个体
            current_gen_fitness = -pop.get_f()
            current_gen_solution = pop.get_x()

            all_fitness.append(current_gen_fitness.copy())
            all_solution.append(current_gen_solution.copy())

            # 更新上一代的最优解
            best_idx = pop.best_idx()
            best_angle, best_loc = pop.get_x()[best_idx]
            problem_instance.update_previous_best(best_angle, best_loc)
        # 👆===========  进行优化  =============👆

        # 获取最优解的目标函数值和决策变量值
        best_fitness = pop.get_f()[pop.best_idx()]

        # ===========  筛选ED最小的形态  ============
        # 展平嵌套列表
        flat_all_fit = [item for sublist in all_fitness for item in sublist]
        flat_all_sol = [item for sublist in all_solution for item in sublist]
        data = pd.DataFrame({
            'Fitness': flat_all_fit,
            'Solution': flat_all_sol,
        })
        # 展开 Fitness 列
        data['Fitness'] = data['Fitness'].apply(lambda x: x[0])
        # 获取最小的 Fitness 值
        min_fitness = data['Fitness'].min()
        # 筛选出 Fitness 等于最小值的行
        min_fitness_solutions = data[data['Fitness'] == min_fitness]['Solution'].tolist()
        # 分别提取 sd_angle 和 sd_location
        sd_angles, sd_locations = [sol[0] for sol in min_fitness_solutions], [sol[1] for sol in min_fitness_solutions]
        for i in range(len(sd_angles)):
            sd_angles[i] = int(mt.degrees(sd_angles[i]))
            sd_locations[i] = round(sd_locations[i], 2)
        ED_list = []
        # 输出结果
        for i in range(len(sd_angles)):
            ED_moment = calculateED.GetPvsdED(best_angle, best_loc, sd_angles[i], sd_locations[i])
            ED_list.append(ED_moment)

        df_ED = pd.DataFrame({
            'Angle': sd_angles,
            'Location': sd_locations,
            'ED': ED_list
        })
        # 筛选出 ED 等于最小值的所有行
        best_ED = min(ED_list)
        all_best_angle = df_ED[df_ED['ED'] == best_ED]['Angle'].values
        all_best_loc = df_ED[df_ED['ED'] == best_ED]['Location'].values

        # 标准化数据
        time_sd_angle = round(all_best_angle[0])
        time_sd_position = all_best_loc[0].round(2)
        # print(df_ED)

        # # 将新的数据保存到文件中，覆盖之前的数据
        # best_value = [time_sd_angle, time_sd_position]
        # with open('saved_values.pkl', 'wb') as ff:
        #     pickle.dump(best_value, ff)
        # print("Saved new values:", best_value)
        #
        # # 数据输出
        # print('best_ED:', best_ED)
        # print('time_sd_angle:', time_sd_angle)
        # print('time_sd_position:', time_sd_position)
        # ===========  筛选ED最小的形态  ============

        return hoy, best_fitness, time_sd_angle, time_sd_position, all_fitness

    @staticmethod
    def main_single(optimize_weight, single_hoy):
        """
        针对单个 HOY 调用优化方法，并显示优化进度。
        """
        print(f"Starting optimization for HOY {single_hoy}...")
        hoy, best_fitness, time_sd_angle, time_sd_position, all_fitness = \
            shade_pygmo.optimize_hoy(single_hoy, epw_dataset, optimize_weight)

        # 输出结果
        print(f"Optimization complete for HOY {hoy}.")
        print(f"Fitness: {best_fitness.round(2)}")
        print(f"Best sd_angle: {time_sd_angle}")
        print(f"Best sd_location: {time_sd_position}")

        return time_sd_angle, time_sd_position

    @staticmethod
    def main_parallel(optimize_weight, hoy_list):

        # 导入数据集
        epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)

        # 并行优化多个HOY
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(shade_pygmo.optimize_hoy, hoy, epw_dataset, optimize_weight) for hoy in hoy_list]
            results = [f.result() for f in as_completed(futures)]

        # 按照 hoy 字段从小到大排序
        sorted_results = sorted(results, key=lambda x: x[0])

        # 输出结果
        for hoy, best_fitness, sd_angle, sd_site, all_fitness in sorted_results:
            print(f"Hoy: {hoy}")
            print(f"Best Fitness: {best_fitness.round(2)}")
            print(f"Best sd_angle: {sd_angle:.2f}")
            print(f"Best sd_location: {sd_site:.2f}")
            print('----------------------------')

        # 汇总所有HOY的最优角度，生成schedule
        schedule = {hoy: (sd_angle, sd_site, best_fitness[0]) for hoy, best_fitness, sd_angle, sd_site, all_fitness in
                    sorted_results}
        hoy_list = list(schedule.keys())
        angles = [values[0] for values in schedule.values()]
        sites = [values[1] for values in schedule.values()]
        best_fitness = [values[2] for values in schedule.values()]

        # # 可视化结果
        visualizeFitness(results, hoy_list)

        # 使用字典创建 DataFrame
        schedule_df = pd.DataFrame({
            'HOY': hoy_list,
            'angle': angles,
            'site': sites,
            'best_fitness': best_fitness,
        }).reset_index(drop=True)

        print("Schedule DataFrame:")
        print(schedule_df)

    @staticmethod
    def outputCSV():
        # 展平嵌套列表
        flat_list = [item for sublist in all_data for item in sublist]
        # 转换为 DataFrame
        my_df = pd.DataFrame(flat_list)
        # 更新代数/个体列表
        UpdateGenAndInd(my_df, pygmo_gen, pygmo_pop, main_hoy)
        dataSaver.save_dataframe(my_df, schedule_name, 'csv')


def main():
    # ===== 输入值 =====
    # >>> 权重输入值 <<<
    my_weights = [weight_dgp, weight_udi, weight_vis, weight_pvg]  # 权重集合
    # ===== 计时器 =====
    start_time = time.time()

    # # >>> 主程序 <<<

    # 👇运算多个Hoy用这个👇
    # 初始化 HOY 总进度条
    with tqdm(total=len(main_hoy), desc="HOY Progress", dynamic_ncols=True) as pbar:
        for hoy in main_hoy:
            # 更新描述为当前 HOY
            pbar.set_description(f"Optimizing HOY {hoy}")

            # 调用单个 HOY 优化方法，带有优化代数的进度条
            schedule = shade_pygmo.main_single(my_weights, hoy)

            # 更新数据到 shade_schedule 中
            main_hoy_list = main_hoy.tolist()
            shade_schedule.loc[main_hoy_list.index(hoy)] = [hoy, schedule[0], schedule[1]]

            # 更新 HOY 总进度条
            pbar.update(1)

    # 👇运算单个Hoy用这个👇
    # shade_pygmo.main_single(my_weights, single_hoy=8)

    # 并行运算目前有点问题，后续考虑优化迭代
    # shade_pygmo.main_parallel(my_weights, main_hoy)

    # ===== 计时器 =====
    end_time = time.time()
    print(f"All HOY optimizations completed in {end_time - start_time:.2f} seconds.")
    # ===== 计时器 =====
    print(shade_schedule)
    shade_schedule.to_csv(schedule_output_path)
    print('done!')
    shade_pygmo.outputCSV()


if __name__ == "__main__":
    main()
