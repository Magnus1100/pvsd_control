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

# 全局变量
fmu = load_fmu("idf_test0704.fmu")

# 初始化结果列表
results = {
    'HOY': [],
    'Cooling Load': [],
    'Heating Load': [],
    'Lighting Energy': [],
    'Shading Angle': [],
    'Shading Position': []
}


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
    fmu

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
