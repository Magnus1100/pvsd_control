import pandas as pd
import matplotlib.pyplot as plt
from fmpy import *
from datetime_gen import timedelta

# 全局变量
my_fmu_path = 'OverhangTest241104.fmu'  # 替换为你的FMU文件路径
my_start_time = 0  # 起始时间，单位为秒
my_stop_time = 86400  # 结束时间（24小时），单位为秒
my_step_size = 3600  # 时间步长为1小时（3600秒）
my_output_variable = 'cooling_load'  # FMU模型中的冷却负荷变量名称


# 加载和运行FMU模型
def run_fmu_simulation(fmu_path, start_time, stop_time, step_size, output_variable):
    # 逐时模拟时间设置my_stop_time
    results = simulate_fmu(fmu_path,
                           start_time=start_time,
                           stop_time=stop_time,
                           step_size=step_size,
                           output=[output_variable])
    # 将模拟结果转换为DataFrame
    df = pd.DataFrame(results)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    return df


def main():
    # 计算结果
    df = run_fmu_simulation(my_fmu_path, my_start_time, my_stop_time, my_step_size, my_output_variable)

    # 绘制冷却负荷逐时变化图像
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df[my_output_variable], label="Cooling Load (W)")
    plt.xlabel('Time')
    plt.ylabel('Cooling Load (W)')
    plt.title('Hourly Cooling Load')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
