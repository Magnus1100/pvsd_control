import pandas as pd

window_name = 'WINDOW_C6660828'
shading_schedule_name = 'SHADING_SCHEDULE-bj250106'

origin_shading_ratio_path = rf'bj-250106/ori_sr.txt'
shading_ratio_df = pd.read_csv(rf'bj-250106/beijing_1111_schedule_20250105.csv')


# Hoy和SR数据（从文件读取）
annual_hoy = pd.DataFrame(range(1, 8761), columns=['Hoy'])
ori_sr = pd.read_csv(origin_shading_ratio_path)
origin_shading = pd.concat([annual_hoy, ori_sr], axis=1)  # 将两个列合并到同一个 DataFrame.
print(origin_shading)

hoy = shading_ratio_df['Hoy']
shading_ratio = shading_ratio_df['Shaded_Ratio']
new_shading = pd.concat([hoy, shading_ratio], axis=1)  # 将两个列合并到同一个 DataFrame.
print(new_shading)

# 合并数据
merged = origin_shading.merge(new_shading, on='Hoy', how='left')

# 用 Shading_Percent 更新 ori_shading_percent，保留原始值为 0 的部分
merged['WINDOW_C6660828'] = merged['Shaded_Ratio'].fillna(merged['WINDOW_C6660828'])

# 删除临时列（如果不需要）
merged = merged.drop(columns=['Hoy'])
merged = merged.drop(columns=['Shaded_Ratio'])
merged.rename(columns={'ori_shading_percent': f'{window_name}'}, inplace=True)
merged.to_csv(f'{shading_schedule_name}.csv', index=False)
