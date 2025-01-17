import pandas as pd
import pvg_calculate as pc
import blind_shade_calculate as bsc

epw = pd.read_csv('source/data/data_shadeCalculate/bj/epwData_bj_withPVG.csv')
panel_area = 4.896
P_stc = 100
shading_schedule = pd.read_csv('shading_schedule/bj-1100-250114/bj_1100_schedule-250117_withVIS.csv')

shading_angle = shading_schedule['SD_Angle']
shading_position = shading_schedule['SD_Position']
hoy = shading_schedule['Hoy']
pvg = []

for i in range(0, len(hoy)):
    hor_angle = epw['Hor_Angle'][i]
    ver_angle = epw['Ver_Angle'][i]
    sd_interval = (0.15 - abs(shading_position[i])).round(2)  # 间距
    shading_percent = bsc.ShadeCalculate.AllShadePercent(2.1, 0.15, sd_interval, hor_angle, ver_angle,
                                                         shading_angle[i])
    surface_radiant = pc.pvgCalculator.calculateRadiant(shading_angle[i], hoy[i])
    surface_pvg = pc.pvgCalculator.calculateHoyPVGeneration(hoy[i], surface_radiant, panel_area, P_stc)
    print(shading_percent)
    pvg_true = surface_pvg * (1 - shading_percent)
    pvg.append(pvg_true)

shading_schedule['PVG'] = pvg
output_path = 'shading_schedule/bj-1100-250114/bj_1100_schedule-250117_withPVG.csv'
shading_schedule.to_csv(output_path, index=False)

print(f"整合完成，文件已保存为 {output_path}")
