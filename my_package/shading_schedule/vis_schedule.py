import pandas as pd

# 读取数据
shading_schedule = pd.read_csv('bj-1100-250114/bj_1100_schedule-250117.csv')
new_shading_schedule_path = 'bj-1100-250114/bj_1100_schedule-250117_withVIS.csv'
shading_angle = shading_schedule['SD_Angle']
shading_position = shading_schedule['SD_Position']

vis_data = pd.read_csv('../source/data/data_shadeCalculate/vis_data_outside_0920.csv')

# 生成 vis_schedule
vis_schedule = []
for i in range(len(shading_position)):
    vis = vis_data[(vis_data['sd_angle'] == shading_angle[i]) &
                   (vis_data['sd_position'] == shading_position[i])]['vis'].values
    vis = vis / 100
    vis_schedule.append(vis[0] if len(vis) > 0 else None)  # 确保有值时取第一个，没值时为 None

# 将 vis_schedule 加入到 shading_schedule
shading_schedule['vis'] = vis_schedule

# 保存结果到新文件（可选）
shading_schedule.to_csv(new_shading_schedule_path, index=False)

print(f"VIS计算完成，文件已保存为 {new_shading_schedule_path}")
