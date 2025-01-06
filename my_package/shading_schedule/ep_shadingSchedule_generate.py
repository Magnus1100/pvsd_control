import pandas as pd

window_name = 'WINDOW_BF23BD9D'
shading_schedule_name = 'SHADING_SCHEDULE-bj240106'

origin_shading_ratio_path = rf'sz-241202/ori_sr.txt'
shading_ratio_path = rf'sz-241202/window_shading_schedule.txt'

# Hoy和SR数据（从文件读取）
annual_hoy = pd.DataFrame(range(1, 8761), columns=['Hoy'])
ori_sr = pd.read_csv(origin_shading_ratio_path)
origin_shading = pd.concat([annual_hoy, ori_sr], axis=1)  # 将两个列合并到同一个 DataFrame.
print(origin_shading)

hoy = pd.read_csv('sz-241202/annual_hoy.txt')
shading_ratio = pd.read_csv(shading_ratio_path)
new_shading = pd.concat([hoy, shading_ratio], axis=1)  # 将两个列合并到同一个 DataFrame.
print(new_shading)

# 合并数据
merged = origin_shading.merge(new_shading, on='Hoy', how='left')

# 用 Shading_Percent 更新 ori_shading_percent，保留原始值为 0 的部分
merged['ori_shading_percent'] = merged['Shading_Percent'].fillna(merged['ori_shading_percent'])

# 删除临时列（如果不需要）
merged = merged.drop(columns=['Shading_Percent'])
merged = merged.drop(columns=['Hoy'])
merged.rename(columns={'ori_shading_percent': f'{window_name}'}, inplace=True)
merged.to_csv(f'{shading_schedule_name}.csv', index=False)
