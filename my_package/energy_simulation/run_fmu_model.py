import math

from pyfmi import load_fmu
import matplotlib.pyplot as plt
import pandas as pd

# 假设您有以下非连续的 HOY 列表和相应的遮阳角度和位置列表
hoy_list = [1, 5, 12, 24, 48, 72, 120, 8760]  # 示例非连续HOY列表
sa_schedule = [30, 45, 60, 75, 90, 105, 120, 135]  # 对应的遮阳角度列表
sd_schedule = [-0.14, -0.13, -0.12, -0.11, -1, 0, 1]  # 对应的遮阳位置列表

# 日程表
schedule_mapping = {
    hoy: {'sa': sa, 'sd': sd}
    for hoy, sa, sd in zip(hoy_list, sa_schedule, sd_schedule)
}

# 初始化结果列表
results = {
    'HOY': [],
    'Cooling Load': [],
    'Heating Load': [],
    'Lighting Energy': [],
    'Shading Angle': [],
    'Shading Position': []
}


class CalculateBlindAxis:
    @staticmethod
    def CalculatePoints(p1, p2, shade_length, shade_angle):
        shade_angle = math.radians(shade_angle)
        y_changed = shade_length * math.sin(shade_angle)
        z_changed = shade_length * math.cos(shade_angle)

        point1 = [p1[0], p1[1], p1[2]]
        point2 = [p1[0], p1[1] + y_changed, p1[2] + z_changed]
        point3 = [p2[0], p1[1] + y_changed, p2[2] + z_changed]
        point4 = [p2[0], p1[1], p1[2]]

        point_list = [point1, point2, point3, point4]
        return point_list

    @staticmethod
    def GetBlindAxis(point_list, blind_count, shade_position):
        blind_axis = []
        shade_interval = 0.15 - abs(shade_position)

        for i in range(blind_count):
            adjusted_points = []
            for point in point_list:
                new_point = [point[0], point[1], point[2]-shade_interval*i]
                adjusted_points.append(new_point)
        blind_axis.append(adjusted_points)


def simResultsShow(df):
    # 绘制逐时冷负荷变化
    plt.plot(df['HOY'], df['Cooling Load'], label='Cooling Load')
    plt.plot(df['HOY'], df['Heating Load'], label='Heating Load')
    plt.xlabel('Hour of Year (HOY)')
    plt.ylabel('Load (W)')
    plt.title('Hourly Cooling and Heating Loads')
    plt.legend()
    plt.show()


def runFmu(fmu_hoy_list):
    # 加载FMU
    fmu = load_fmu("idf_test0704.fmu")

    # 逐时仿真并记录结果
    for hoy in fmu_hoy_list:  # 一年共有8760个小时
        # 查找当前HOY对应的遮阳角度和位置
        shading_angle = schedule_mapping[hoy]['sa']
        shading_position = schedule_mapping[hoy]['sd']

        fmu.set("shading_angle", shading_angle)
        fmu.set("shading_position", shading_position)

        # 运行FMU仿真
        res = fmu.simulate(start_time=hoy, stop_time=hoy + 1)

        # 存储仿真结果
        results.append({
            "HOY": hoy,
            "Cooling Load": res['cooling_load_variable'],
            "Heating Load": res['heating_load_variable'],
            "Lighting Energy": res['lighting_energy_variable'],
            "Shading Angle": shading_angle,
            "Shading Position": shading_position
        })

    # 将结果转换为Pandas DataFrame
    df = pd.DataFrame(results)

    # 输出结果为CSV文件
    df.to_csv("simulation_results.csv", index=False)

    # 或者直接查看DataFrame内容
    print(df.head())


def main():
    runFmu(hoy_list)


if __name__ == '__main__':
    main()
