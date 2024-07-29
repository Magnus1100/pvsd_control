import time
import math as mt
import numpy as np
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
import joblib

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from analytic_formula import blind_shade_calculate as bsc
from analytic_formula import pvg_calculate as pc

# 声明 pvsd 实例
pvsd_instance = bsc.pvShadeBlind(0.15, 2.1, 20, 0.7, 0)


class hoyEditor:
    @staticmethod
    def generateHoyList(start_date_str, end_date_str, exclude_weekends=False, start_hour=0, end_hour=23):
        """
        根据输入的日期范围和条件生成 HOY 列表。

        Args:
            start_date_str (str): 开始日期，格式为 'MM-DD'。
            end_date_str (str): 结束日期，格式为 'MM-DD'。
            exclude_weekends (bool): 是否排除周末，默认为 False。
            start_hour (int): 每天的开始小时，默认为 0。
            end_hour (int): 每天的结束小时，默认为 23。

        Returns:
            list: 对应的 HOY 列表。
        """
        # 获取当前年份
        current_year = datetime.now().year

        # 使用当前年份解析开始日期和结束日期
        start_date = datetime.strptime(f"{current_year}-{start_date_str}", '%Y-%m-%d')
        end_date = datetime.strptime(f"{current_year}-{end_date_str}", '%Y-%m-%d')

        hoy_list = []
        current_date = start_date

        while current_date <= end_date:
            if exclude_weekends and current_date.weekday() >= 5:  # 排除周末
                current_date += timedelta(days=1)
                continue

            for hour in range(start_hour, end_hour + 1):
                days_since_start_of_year = (current_date - datetime(current_date.year, 1, 1)).days
                hoy = days_since_start_of_year * 24 + hour
                hoy_list.append(hoy)

            current_date += timedelta(days=1)

        return hoy_list


def visualizeFitness(results, hoy_list):
    """
    展示不同HOY的平均适应度值随代数变化的图表。

    参数:
    - results (list): 包含每个HOY优化结果的元组列表。
    - hoy_list (list): 被优化的HOY列表。
    """
    cmap = plt.get_cmap('viridis')  # 获取viridis色图
    colors = cmap(np.linspace(0.2, 0.8, len(hoy_list)))  # 生成同一色系的不同深浅颜色

    for i, (hoy, _, _, _, all_fitness) in enumerate(results):
        avg_fitness_per_generation = []

        # 计算每代的平均 fitness
        for gen_fitness in all_fitness:
            avg_fitness = np.mean(gen_fitness)
            avg_fitness_per_generation.append(avg_fitness)

        # 绘制每代的折线图
        generations = np.arange(len(avg_fitness_per_generation))
        plt.plot(generations, avg_fitness_per_generation, marker='o', color=colors[i], label=f"Hoy {hoy}")

    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Values Over Generations for Different HOYs')
    plt.legend()
    plt.show()


class calculateED:
    @staticmethod  # 根据遮阳状态计算相对坐标
    def GetAxis(sd_angle, sd_location, sd_index, hw):
        # 角度转弧度
        sd_angle = mt.radians(sd_angle)

        # 计算坐标（相对坐标）
        x_axis = mt.sin(sd_angle)
        if sd_location >= 0:
            z_axis = hw - pvsd_instance.sd_width * (1 - mt.cos(sd_angle)) + sd_location * sd_index
        else:
            z_axis = hw - pvsd_instance.sd_width * (1 - mt.cos(sd_angle)) + sd_location * (sd_index + 1)
        return x_axis, z_axis

    @staticmethod  # 根据坐标计算欧式距离
    def GetED(a_origin, a_next, b_origin, b_next):
        ED = mt.sqrt(mt.pow(a_next - a_origin, 2) + mt.pow(b_next - b_origin, 2))
        return ED

    @staticmethod  # 计算两个坐标的欧式距离
    def GetPvsdED(angle_origin, loc_origin, angle_next, loc_next, shade_count):
        # 总欧氏距离
        total_ED = 0

        # 每个版单独计算
        for i in range(1, shade_count + 1):
            x_origin, z_origin = calculateED.GetAxis(angle_origin, loc_origin, i, pvsd_instance.sd_width)  # 原始坐标
            x_next, z_next = calculateED.GetAxis(angle_next, loc_next, i, pvsd_instance.sd_width)  # 下个状态坐标

            ED_i = calculateED.GetED(x_origin, x_next, z_origin, z_next)  # 计算总欧式距离
            total_ED += ED_i
        return total_ED


class MyProblem:
    def __init__(self, hoy, azimuth, altitude, ver_angle, hor_angle, my_weights, max_pvg):
        self.n_var = 2  # 两个变量：sd_interval, sd_angle
        self.n_obj = 1  # 单目标优化

        self.hoy = hoy
        self.Azimuth = azimuth
        self.Altitude = altitude
        self.ver_angle = ver_angle
        self.hor_angle = hor_angle
        self.my_weights = my_weights
        self.max_pvg = max_pvg
        self.fitness_history = []  # 保存每一步的适应度

    def fitness(self, x):
        sd_angle, sd_location = x
        sd_location = sd_location.round(2)
        sd_interval = (0.12 - abs(sd_location)).round(2)  # 间距
        sd_angle_degree = int(mt.degrees(sd_angle))

        # 调用外部的机器学习模型进行预测
        predict_parameter = [self.Azimuth, self.Altitude, sd_angle_degree, sd_location]
        feature_names = ['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']
        predict_parameters = pd.DataFrame([predict_parameter], columns=feature_names)

        model_sdgp = joblib.load('./source/model_optimizer/sDGP_random_forest_model.pkl')
        model_sudi = joblib.load('./source/model_optimizer/sUDI_random_forest_model.pkl')

        pvsd = pvsd_instance
        """
        计算各优化值
        - sdgp(0-1) : 眩光概率>0.38的空间比例，ml预测
        - sudi(0-1) : 日光舒适比例照度[200-800lx]空间比例，ml预测
        - vis(0-1) : 室内平均视野水平，数据库调用
        - pvg_normal(0-1) : 归一化的光伏发电量（计算值/最大值），公式计算
        - weighted_vals : 加权得到的优化值
        """
        val_sdgp = model_sdgp.predict(predict_parameters)[0]
        val_sudi = model_sudi.predict(predict_parameters)[0]

        val_vis = bsc.ShadeCalculate.GetVis(sd_angle_degree, sd_location)
        val_vis = float(val_vis[0])

        shade_percent = bsc.ShadeCalculate.AllShadePercent(pvsd.sd_length, pvsd.sd_width, sd_interval,
                                                           self.ver_angle, self.hor_angle, sd_angle)
        shade_rad = pc.pvgCalculator.calculateIrradiance(pvsd.window_azimuth, sd_angle, 0.6, self.hoy)
        pvg_value = pc.pvgCalculator.calculateHoyPvGeneration(shade_rad, pvsd.panel_area, pvsd.pv_efficiency)
        val_shade = pvg_value * shade_percent / self.max_pvg

        # 加权优化值
        weighted_vals = - (val_sdgp * self.my_weights[0] + val_sudi * self.my_weights[1] + val_vis * self.my_weights[2]
                           + val_shade * self.my_weights[3])

        # 保存每一步的适应度
        self.fitness_history.append(- weighted_vals)

        # ========== 打印结果 ==========
        print('sd_angle: ' + str(sd_angle))
        print('sd_location: ' + str(sd_location))
        print('weighted_vals: ' + str(weighted_vals))
        print('----------------------------')
        # ========== 打印结果 ==========
        return [weighted_vals]

    @staticmethod
    def get_bounds():
        return [mt.radians(0), -0.12], [mt.radians(90), 0.12]

    @staticmethod
    def get_names():
        return ['sd_angle', 'sd_site']


def optimize_hoy(hoy, epw_dataset, my_weights, gen_size=2, pop_size=2):
    my_ver_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Ver_Angle')
    my_hor_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Hor_Angle')
    my_azimuth = epw_dataset.loc[hoy, 'Azimuth']
    my_altitude = epw_dataset.loc[hoy, 'Altitude']
    my_max_pvg = epw_dataset.loc[hoy, 'max_pv_generation']

    problem_instance = MyProblem(hoy, my_azimuth, my_altitude, my_ver_angle, my_hor_angle, my_weights, my_max_pvg)
    prob = pg.problem(problem_instance)

    # 粒子群优化算法
    algo = pg.algorithm(pg.pso(gen=1))
    # 创建种群
    pop = pg.population(prob, size=pop_size)
    # 用于保存所有代的适应度值
    all_fitness = []
    # 进行优化
    for gen in range(gen_size):
        pop = algo.evolve(pop)
        current_gen_fitness = -pop.get_f()  # 获取当前代所有个体的适应度值
        all_fitness.append(current_gen_fitness.copy())

    # 获取最优解的目标函数值和决策变量值
    best_fitness = pop.get_f()[pop.best_idx()]
    best_solution = pop.get_x()[pop.best_idx()]

    time_sd_angle = round(mt.degrees(best_solution[0]))
    time_sd_site = best_solution[1].round(2)

    return hoy, best_fitness, time_sd_angle, time_sd_site, all_fitness


def main_single(optimize_weight):
    # ===== 计时器 =====
    start_time = time.time()
    # ===== 计时器 =====

    # ===== 输入值 =====
    single_hoy = 12
    # ===== 输入值 =====

    # 导入数据集
    epw_data_file_path = './source/data/epw_data.csv'
    epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)

    # 优化单个HOY
    hoy, best_fitness, time_sd_angle, time_sd_site, all_fitness = optimize_hoy(single_hoy, epw_dataset, optimize_weight)

    # ===== 计时器 =====
    end_time = time.time()
    execution_time = format(end_time - start_time, '.2f')
    print("Time cost:", execution_time, "s")
    # ===== 计时器 =====

    # ===== 可视化 =====
    cmap = plt.get_cmap('viridis')  # 获取viridis色图
    colors = cmap(np.linspace(0.2, 0.8, len(all_fitness)))  # 生成同一色系的不同深浅颜色
    for gen in range(len(all_fitness)):
        fitness_values = all_fitness[gen]
        generations = [gen] * len(fitness_values)  # 生成当前代的代数
        plt.scatter(generations, fitness_values, color=colors[gen])

    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Values Over Generations')
    plt.xlim(0, len(all_fitness) - 1)  # 设置横坐标轴的范围，从 0 到 gen_size - 1
    plt.show()
    # ===== 可视化 =====

    # ===== 输出最优个体 =====
    print("Best solution:")
    print("Fitness:", best_fitness.round(2))
    print("Best sd_angle:", time_sd_angle)
    print("Best sd_location:", time_sd_site)
    # ===== 输出最优个体 =====


def main_parallel(optimize_weight):
    # ===== 计时器 =====
    start_time = time.time()
    # ===== 计时器 =====

    # 日期输入值
    start_date = "6-21"
    end_date = "6-21"
    skip_weekdays = False
    start_hour = 8
    end_hour = 17
    hoy_list = hoyEditor.generateHoyList(start_date, end_date, skip_weekdays, start_hour, end_hour)  # 需要优化的HOY列表
    print(hoy_list)
    # ===== 输入值 =====

    # 导入数据集
    epw_data_file_path = './source/data/epw_data.csv'
    epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)

    # 并行优化多个HOY
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(optimize_hoy, hoy, epw_dataset, optimize_weight) for hoy in hoy_list]
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

    # ===== 计时器 =====
    end_time = time.time()
    execution_time = format(end_time - start_time, '.2f')
    print("Total time cost:", execution_time, "s")
    # ===== 计时器 =====

    # 可视化结果
    visualizeFitness(results, hoy_list)

    # 创建 DataFrame
    schedule_df = pd.DataFrame.from_dict(schedule, orient='index', columns=['sd_angle', 'sd_site', 'best_fitness'])
    schedule_df.index.name = 'HOY'

    print("Schedule DataFrame:")
    print(schedule_df)

    # schedule_df = pd.DataFrame({'hoy': hoy in sorted_results})
    # epw_dataset['sd_angle'] = sd_angle in sorted_results
    # epw_dataset['sd_site'] = sd_site in sorted_results
    # epw_dataset['best_fitness'] = best_fitness in sorted_results
    #
    # print(schedule_df)


def main():
    # ===== 输入值 =====
    # 权重输入值
    weight1 = 0.1  # glare_weight[0,1]
    weight2 = 0.4  # daylight_weight[0,1]
    weight3 = 0.4  # visibility_weight[0,1]
    weight4 = 1 - (weight1 + weight2 + weight3)  # pv_generation_weight[0,1]
    my_weights = [weight1, weight2, weight3, weight4]  # 权重

    print("请选择要运行的优化方式:")
    print("1. 单个优化")
    print("2. 并行优化")

    choice = input("请输入选项数字 (1 or 2): ")

    if choice == '1':
        main_single(my_weights)
    elif choice == '2':
        main_parallel(my_weights)
    else:
        print("无效的选项，请输入 '1' 或 '2'.")


if __name__ == "__main__":
    main()
