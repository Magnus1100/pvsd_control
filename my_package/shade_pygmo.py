import time
import joblib
import math as mt
import numpy as np
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from analytic_formula import blind_shade_calculate as bsc
from analytic_formula import pvg_calculate as pc

# 声明 pvsd 实例
pvsd_instance = bsc.pvShadeBlind(0.15, 2.1, 20, 0.7, 0,
                                 0.6, 16, 2.4)
epw_data_file_path = 'source/dataset/epw_data.csv'
vis_data = pd.read_csv('source/dataset/vis_data.csv')
sDGP = np.loadtxt('source/data/sDGP.txt')
sUDI = np.loadtxt('source/data/sUDI.txt')


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
    def __init__(self, hoy, azimuth, altitude, ver_angle, hor_angle, my_weights, max_pvg):
        self.n_var = 2  # 两个变量：sd_interval, sd_angle
        self.n_obj = 1  # 单目标优化

        # 存储每一代最优个体
        self.previous_best_angle = None
        self.previous_best_loc = None

        self.hoy = hoy
        self.Azimuth = azimuth
        self.Altitude = altitude
        self.ver_angle = ver_angle
        self.hor_angle = hor_angle
        self.my_weights = my_weights
        self.max_pvg = max_pvg
        self.fitness_history = []  # 保存每一步的适应度
        # self.data_collector = pd.DataFrame(
        #     columns=['sdgp:', 'sdgp_valued:', 'sUDI:', 'sUDI_valued:', 'vis', 'vis_valued:', 'pvg', 'pvg_valued:', 'ED',
        #              'ED_valued:'])

    def fitness(self, x):
        sd_angle, sd_location = x
        sd_location = sd_location.round(2)
        sd_interval = (0.15 - abs(sd_location)).round(2)  # 间距
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
        1, sdgp(0-1) : 眩光概率>0.38的空间比例，ml预测
        2, sudi(0-1) : 日光舒适比例照度[200-800lx]空间比例，ml预测
        3, vis(0-1) : 室内平均视野水平，数据库调用
        4, pvg_normal(0-1) : 归一化的光伏发电量（计算值/最大值），公式计算
        5, ED_percent : 形变的欧式距离，用在一定程度上减少形变
        - weighted_vals : 加权得到的优化值
        """
        # 1，2，sudi/sdgp 机器学习模型预测
        pred_sdgp = model_sdgp.predict(predict_parameters)[0]
        normalized_sdgp = normalizeValue(pred_sdgp, min(sDGP) / 100, max(sDGP) / 100)  # 标准化sDGP(sDGP最小值为76，需要标准化)
        val_sdgp = normalized_sdgp * self.my_weights[0]

        pred_sudi = model_sudi.predict(predict_parameters)[0]
        val_sudi = pred_sudi * self.my_weights[1]

        # 3，数据库调用查询vis
        vis = bsc.ShadeCalculate.GetVis(sd_angle_degree, sd_location)
        vis = float(vis[0])
        val_vis = vis * self.my_weights[2]

        # 4，调用公式计算pv发电量
        shade_percent = bsc.ShadeCalculate.AllShadePercent(pvsd.sd_length, pvsd.sd_width, sd_interval, self.ver_angle,
                                                           self.hor_angle, sd_angle)
        shade_rad = pc.pvgCalculator.calculateIrradiance(pvsd.window_azimuth, sd_angle, 0.6, self.hoy)
        pvg_value = pc.pvgCalculator.calculateHoyPvGeneration(shade_rad, pvsd.panel_area, pvsd.pv_efficiency)
        normalized_pvg = pvg_value * shade_percent / self.max_pvg
        val_pvg = normalized_pvg * self.my_weights[3]

        # 5，加入形变参数
        ED_max = calculateED.GetPvsdED(0, 0.15, 90, -0.15)
        if self.previous_best_angle is None or self.previous_best_loc is None:
            ED_moment = calculateED.GetPvsdED(0, 0.15, sd_angle_degree, sd_location)
        else:
            ED_moment = calculateED.GetPvsdED(self.previous_best_angle, sd_angle_degree, self.previous_best_loc,
                                              sd_location)
        normalized_ED = ED_moment / ED_max
        val_ED = normalized_ED * 0.1  # 权重设置为0.1

        # final value - 加权优化值
        val_all = val_sdgp + val_sudi + val_vis + val_pvg + val_ED
        val_optimize = - val_all

        # 保存每一步的适应度
        self.fitness_history.append(- val_optimize)

        # 保存每一步值，以便分析
        # self.data_collector = self.data_collector.append({
        #     'sdgp:': pred_sdgp,
        #     'sdgp_valued:': val_sdgp,
        #     'sUDI:': pred_sudi,
        #     'sUDI_valued:': val_sudi,
        #     'vis': vis,
        #     'vis_valued:': val_vis,
        #     'pvg': pvg_value,
        #     'pvg_valued:': val_pvg,
        #     'ED': ED_moment,
        #     'ED_valued:': val_ED
        # }, ignore_index=True)

        # ========== 打印结果 ==========
        print('sdgp: %.2f' % pred_sdgp)
        print('sudi: %.2f' % pred_sudi)
        print('vis: %.2f' % vis)
        print('pvg: %.2f' % pvg_value)
        print('ED: %.2f' % ED_moment)
        print('---------------------------')
        print('val_sdgp: %.2f' % val_sdgp)
        print('val_sudi: %.2f' % val_sudi)
        print('val_vis: %.2f' % val_vis)
        print('val_pvg: %.2f' % val_pvg)
        print('val_ED: %.2f' % val_ED)
        print('----------------------------')
        print('sd_angle: ' + str(sd_angle_degree))
        print('sd_location: ' + str(sd_location))
        print('weighted_vals: %2f' % abs(val_optimize))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # ========== 打印结果 ==========
        return [val_optimize]

    @staticmethod
    def get_bounds():
        return [mt.radians(0), -0.12], [mt.radians(90), 0.12]

    @staticmethod
    def get_names():
        return ['sd_angle', 'sd_site']

    # 更新上一代最优个体
    def update_previous_best(self, angle, loc):
        self.previous_best_angle = angle
        self.previous_best_loc = loc


class shade_pygmo:
    @staticmethod
    def optimize_hoy(hoy, epw_dataset, my_weights, gen_size=10, pop_size=10):
        my_ver_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Ver_Angle')
        my_hor_angle = bsc.ShadeCalculate.GetAngle(hoy, 'Hor_Angle')
        my_azimuth = epw_dataset.loc[hoy, 'Azimuth']
        my_altitude = epw_dataset.loc[hoy, 'Altitude']
        my_max_pvg = epw_dataset.loc[hoy, 'max_pv_generation']

        # 创建一个空的 DataFrame，用于存储所有 HOY 的 data_collector 数据
        # all_data_collectors = pd.DataFrame()

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

            # 获取当前代最优个体
            best_idx = pop.best_idx()
            best_angle, best_loc = pop.get_x()[best_idx]
            problem_instance.update_previous_best(best_angle, best_loc)  # 更新上一代最优个体
            # all_data_collectors = pd.concat([all_data_collectors, problem_instance.data_collector], ignore_index=True)

        # 获取最优解的目标函数值和决策变量值
        best_fitness = pop.get_f()[pop.best_idx()]
        best_solution = pop.get_x()[pop.best_idx()]

        time_sd_angle = round(mt.degrees(best_solution[0]))
        time_sd_site = best_solution[1].round(2)
        # all_data_collectors.to_csv('all_data_collectors.csv')

        return hoy, best_fitness, time_sd_angle, time_sd_site, all_fitness

    @staticmethod
    def main_single(optimize_weight, single_hoy):
        # ===== 计时器 =====
        start_time = time.time()

        # 导入数据集
        epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)
        # ===== 输入值 =====

        # 优化单个HOY
        hoy, best_fitness, time_sd_angle, time_sd_site, all_fitness = shade_pygmo.optimize_hoy(single_hoy, epw_dataset,
                                                                                               optimize_weight)

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

    @staticmethod
    def main_parallel(optimize_weight, hoy_list):
        # ===== 计时器 =====
        start_time = time.time()
        # ===== 计时器 =====

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


def main():
    # ===== 输入值 =====
    # 权重输入值
    weight_dgp = 0  # 眩光权重[0,1]
    weight_udi = 0  # 采光权重[0,1]
    weight_vis = 0  # 视野权重[0,1]
    weight_pvg = 1  # 光伏发电量权重[0,1]
    my_weights = [weight_dgp, weight_udi, weight_vis, weight_pvg]  # 权重集合

    # 生成 hoy 列表
    start_date = "6-21"  # 夏至日
    end_date = "6-21"
    skip_weekdays = False
    start_hour = 8
    end_hour = 17
    main_hoy = hoyEditor.generateHoyList(start_date, end_date, skip_weekdays, start_hour, end_hour)  # 需要优化的HOY列表

    if isinstance(main_hoy, list):
        shade_pygmo.main_parallel(my_weights, main_hoy)
    else:
        shade_pygmo.main_single(my_weights, main_hoy)


if __name__ == "__main__":
    main()
