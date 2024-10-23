import joblib
import pandas as pd
import math as mt
import numpy as np
import shade_optimizer as so

model_sdgp = joblib.load('./source/model_optimizer/model_0920/sDGP_RF_0920.pkl')
model_sudi = joblib.load('./source/model_optimizer/model_0920/sUDI_RF_0920.pkl')

epw_data_file_path = r'./source/dataset/epw_data.csv'
epw_dataset = pd.read_csv(epw_data_file_path, index_col=0)
Azimuth = np.loadtxt('./source/data/azimuth.txt')
Altitude = np.loadtxt('./source/data/altitude.txt')

# 取值范围
min_angle, max_angle = mt.radians(0), mt.radians(90)  # 角度范围
min_position, max_position = -0.14, 0.14  # 位置范围
min_azimuth, max_azimuth = mt.radians(min(Azimuth)), mt.radians(max(Azimuth))  # 方位角范围
min_altitude, max_altitude = mt.radians(min(Altitude)), mt.radians(max(Altitude))  # 高度角范围

pred_sDGP_data = []
pred_sUDI_data = []

hoys = np.loadtxt('./source/data/hoys.txt')
orientation = -90
angle = 90
position = 0

# 角度
normalized_angle = so.normalizeValue(angle, min_angle, max_angle)

# 位置
normalized_position = so.normalizeValue(position, min_position, max_position)

for hoy in hoys:
    # 方位角
    real_azimuth = epw_dataset.loc[hoy, 'Azimuth']
    amend_azimuth = real_azimuth + orientation
    azimuth = mt.radians(amend_azimuth)
    normalized_azimuth = so.normalizeValue(amend_azimuth, min_azimuth, max_azimuth)

    # 高度角
    real_altitude = epw_dataset.loc[hoy, 'Altitude']
    altitude = mt.radians(real_altitude)
    normalized_altitude = so.normalizeValue(altitude, min_altitude, max_altitude)

    feature_names = ['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']
    predict_parameter = [normalized_azimuth, normalized_altitude, normalized_angle, normalized_position]
    predict_parameters = pd.DataFrame([predict_parameter], columns=feature_names)

    pred_sdgp = model_sdgp.predict(predict_parameters)[0]
    pred_sudi = model_sudi.predict(predict_parameters)[0]

    pred_sDGP_data.append(pred_sdgp)
    pred_sUDI_data.append(pred_sudi)

    print("pred_sdgp: " + str(pred_sdgp))
    print("pred_sudi: " + str(pred_sudi))

# 定义数组
my_array = [1, 2, 3, 4, 5]

# 以写入模式打开文件
with open('ori_verify_sDGP.txt', 'w') as f:
    # 遍历数组，将每个元素写入文件，并添加换行符
    for item in pred_sDGP_data:
        f.write(f"{item}\n")

# 以写入模式打开文件
with open('ori_verify_sUDI.txt', 'w') as f:
    # 遍历数组，将每个元素写入文件，并添加换行符
    for item in pred_sUDI_data:
        f.write(f"{item}\n")
