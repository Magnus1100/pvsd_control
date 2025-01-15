from datetime import datetime, timedelta
import pandas as pd

# 示例 HOY 列表，可以替换为你的实际数据
schedule = pd.read_csv('shading_schedule/bj-1111-250106/shading_schedule_withPVG.csv')
hoy_list = schedule['Hoy']

# 基准日期，一般是当年的第一天（UTC 时间）
base_date = datetime(2024, 1, 1)
dt_list = []

for hoy in hoy_list:
    dt = base_date + timedelta(hours=hoy - 1)
    dt_list.append(dt)

schedule['Date'] = dt_list

schedule.to_csv('shading_schedule_withPVG.csv', index=False)