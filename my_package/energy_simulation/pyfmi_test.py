import pyfmi
import pprint
import pandas as pd
import overhang_calculate as oc
import numpy as np
import matplotlib.pyplot as plt
import csv

shade_schedule = {
    'hoy': [1, 2, 3, 4, 5, 6, 7, 8],
    'sd_angle': [90, 90, 90, 90, 90, 90, 90, 90],
    'sd_position': [0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07]
}
shade_schedule_df = pd.DataFrame(shade_schedule)

model_name = 'OverhangTest241104'

# 设置参数
days = 365  # 一年
hours = 24  # 每天 24 小时
minutes = 60  # 每小时 60 分钟
seconds = 60  # 每分钟 60 秒

# 计算总步数
num_steps = days * hours  # 总步数
time_stop = days * hours * minutes * seconds
second_step = 3600  # 每个时间步长是 1 小时（3600 秒）

print("Time Step Length (Seconds):", second_step)

# 加载fmu模型
model = pyfmi.load_fmu(model_name + '.fmu')
opts = model.simulate_options()
opts['ncp'] = num_steps  # 设置模拟步数
opts['initialize'] = False

# 初始化模型
sim_time = 0
# model.initialize(sim_time, time_stop)
a_hoy = 1

# 遮阳角度
hoy_sd_angle = shade_schedule_df[shade_schedule_df['hoy'] == a_hoy]['sd_angle'][0]
input_shade_angle = ('TiltAngleFromWindowDoor', hoy_sd_angle)

# 遮阳位置
hoy_sd_position = shade_schedule_df[shade_schedule_df['hoy'] == a_hoy]['sd_position'][0]
OverhangPosition = oc.GetOverhangParameters(hoy_sd_position, sd_count=16, window_height=2.4)
pprint.pprint(OverhangPosition)
input_shade_position = ('HeightabovewindoworDoor', OverhangPosition)

index = 0
input_check_angle, input_check_position = [], []
Heating_Load, Cooling_load, Lighting_load = [], [], []

while sim_time < time_stop:
    for i in range(1, len(input_shade_position) + 1):  # 假设 i 从 1 开始
        model.set(f'PVSDSLA{i}.TiltAngleFromWindowDoor', input_shade_angle)
        model.set(f'PVSDSLA{i}.HeightabovewindoworDoor', input_shade_position[i - 1])

        input_check_angle[index] = model.get(f'PVSDSLA{i}.TiltAngleFromWindowDoor')
        input_check_position[index] = model.get(f'PVSDSLA{i}.HeightabovewindoworDoor')

    # 进行一个时间步长的模拟
    res = model.do_step(current_t=sim_time, step_size=second_step, new_step=True)

    sim_time += second_step
    index += 1


