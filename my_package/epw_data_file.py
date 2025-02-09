import math
import numpy as np
import pandas as pd

'''
！！！需要修改的内容
地区：【深圳 = sz】|【北京 = bj】|【昆明 = km】|【哈尔滨 = hb】
要改：
1. aim_location
2. 主程序里用的经纬和时区，在最⬇️
'''

aim_location = 'sz'
location_bj = [39.93, 116.28, 8.0]
location_sz = [22.55, 114.10, 8.0]
location_hb = [45.75, 126.77, 8.0]
location_km = [25.02, 102.68, 8.0]

epw_data_path = f'./source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}.csv'
hoys_path = f'./source/data/hoys/hoy_{aim_location}/annual_hoy.txt'
output_path = f'./source/data/data_shadeCalculate/{aim_location}/f_Directory_{aim_location}.csv'

# 读取文件
epw_dataset = pd.read_csv(epw_data_path)
epw_dataset.set_index('Hoy', inplace=True)  # 将 'hoy' 列设置为索引
hoys = np.loadtxt(hoys_path)


# 获取f_value
def getFValues(d):
    if 1.000 <= d < 1.065:
        return [-0.008, 0.588, -0.062, -0.060, 0.072, -0.022]
    elif 1.065 <= d < 1.230:
        return [0.130, 0.683, -0.151, -0.019, 0.066, -0.029]
    elif 1.230 <= d < 1.500:
        return [0.330, 0.487, -0.221, 0.055, -0.064, -0.026]
    elif 1.500 <= d < 1.950:
        return [0.568, 0.187, -0.295, 0.109, -0.152, -0.014]
    elif 1.950 <= d < 2.800:
        return [0.873, -0.392, -0.362, 0.226, -0.462, 0.001]
    elif 2.800 <= d < 4.500:
        return [1.132, -1.237, -0.412, 0.288, -0.823, 0.056]
    elif 4.500 <= d < 6.200:
        return [1.060, -1.600, -0.359, 0.264, -1.127, 0.131]
    elif d >= 6.200:
        return [0.678, -0.327, -0.250, 0.156, -1.377, 0.251]
    else:
        return [0] * 6


def calculateSolarParameters(f_hoys, lat, lon, tz):
    """
    计算f1,f2 [f1和f2是环绕太阳和地平线的亮度系数]

    Args:
    - f_hoys (list[int]): 需要计算的 hoy 列表
    - lat (float): 纬度latitude
    - lon (float): 经度longitude
    - tz (float): 时区

    Returns:
    - df_f1_f2 :'f1''f2' dataframe
    """
    b_365, et_365, delta_365, delta_sin_365, delta_cos_365, Gon_365, f1_hoy, f2_hoy = [], [], [], [], [], [], [], []
    calculated_azimuth_hoy = []
    calculated_zenith_hoy = []
    Gsc = 1366.1  # solar constant (W/m2)
    lat_cos = math.cos(math.radians(lat))
    lat_sin = math.sin(math.radians(lat))
    lst = tz * 15

    for n in range(1, 366):
        b = (n - 1.0) * 360.0 / 365.0 * math.pi / 180.0  # B 转化为弧度
        d = 6.24004077 + 0.01720197 * (365.25 * (2024 - 2000) + n)  # wiki公式中的 d 计算
        et_wiki = -7.659 * math.sin(d) + 9.863 * math.sin(2 * d + 3.5932)  # wiki公式计算 et_wiki

        # 计算赤纬（declination）δ
        delta = (0.006918 - 0.399912 * math.cos(b) + 0.070257 * math.sin(b) -
                 0.006758 * math.cos(2 * b) + 0.000907 * math.sin(2 * b) -
                 0.002697 * math.cos(3 * b) + 0.00148 * math.sin(3 * b))

        delta_sin = math.sin(delta)  # sinδ
        delta_cos = math.cos(delta)  # cos

        # 计算地外辐射 Gon
        Gon = Gsc * (1.000110 + 0.034221 * math.cos(b) + 0.001280 * math.sin(b) + 0.00719 * math.cos(
            2 * b) + 0.000077 * math.sin(2 * b))

        # 计算结果加入列表
        b_365.append(b)
        et_365.append(et_wiki)
        delta_365.append(delta)
        delta_sin_365.append(delta_sin)
        delta_cos_365.append(delta_cos)
        Gon_365.append(Gon)

    for h in f_hoys:
        h = int(h)
        solar_time = (h % 24 + 0.5 + et_365[h // 24] / 60 + (lon - lst) / 15 + 24) % 24

        if solar_time < 0:
            omega = 15 * (solar_time + 12)  # Ω 时间角
        else:
            omega = 15 * (solar_time - 12)
        omega_cos = math.cos(math.radians(omega))

        # 天顶角
        zenith_cos = lat_cos * delta_cos_365[h // 24] * omega_cos + lat_sin * delta_sin_365[h // 24]
        zenith = math.acos(zenith_cos)
        zenith_sin = math.sin(zenith)

        # 方位角
        az_init = (lat_sin * zenith_cos - delta_sin_365[h // 24]) / (lat_cos * zenith_sin)
        try:
            if omega > 0:
                calculated_azimuth = (math.acos(az_init) + math.pi) % (2 * math.pi) - math.pi
            else:
                calculated_azimuth = (3 * math.pi - math.acos(az_init)) % (2 * math.pi) - math.pi
        except ValueError:
            calculated_azimuth = 0

        # Epsilon ε a clearness 一种天空透明度指数
        zenithDegree = math.degrees(zenith)
        print(zenithDegree)
        if zenithDegree > 96.07995:  # 96.07995 是一个经验常数，用于太阳天顶角的修正公式。它的物理意义是描述太阳接近地平线时的辐射特性。
            print(f"Warning: zenithDegree {zenithDegree} exceeds 96.07995, may cause math domain error.")

        if epw_dataset["Diffuse_Rad"][h] != 0:
            epsilon = ((((epw_dataset["Direct_Rad"][h] + epw_dataset["Diffuse_Rad"][h]) / epw_dataset["Diffuse_Rad"][
                h]) + (0.000005535 * (zenithDegree ** 3))) / (1 + (0.000005535 * (zenithDegree ** 3))))
        else:
            # 处理分母为零的情况
            epsilon = float('inf')  # 或者设置为其他你认为合适的值

        if zenithDegree > 96.07995:
            brightness = 1e-5
        else:
            brightness = epw_dataset["Diffuse_Rad"][h] / (
                    (zenith_cos + 0.50572 * math.pow(96.07995 - zenithDegree, -1.6364)) * Gon_365[h // 24])
        fValues = getFValues(epsilon)
        F1 = max(0, fValues[0] + fValues[1] * brightness + zenith * fValues[2])
        F2 = fValues[3] + fValues[4] * brightness + zenith * fValues[5]

        f1_hoy.append(F1)
        f2_hoy.append(F2)
        calculated_azimuth_hoy.append(math.degrees(calculated_azimuth))
        calculated_zenith_hoy.append(zenithDegree)

    df_f1_f2 = pd.DataFrame({
        'Hoy': hoys,
        'f1': pd.Series(f1_hoy),
        'f2': pd.Series(f2_hoy),
        'calculated_azimuth': pd.Series(calculated_azimuth_hoy),
        'calculated_zenith': pd.Series(calculated_zenith_hoy),
    })
    df_f1_f2.set_index('Hoy', inplace=True)

    return df_f1_f2


def main():
    f_directory = calculateSolarParameters(hoys, lat=location_km[0], lon=location_km[1], tz=location_km[2])
    f_directory.to_csv(output_path)


if __name__ == '__main__':
    main()
