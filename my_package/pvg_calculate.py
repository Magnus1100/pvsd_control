import math
import ast
import pandas as pd
import numpy as np

#  全局变量
aim_location = 'bj'
single_panel_area, total_panel_area = 0.306, 4.896
pvg_epw_dataset = pd.read_csv(f'./source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}.csv')
pvg_f_directory = pd.read_csv(f'./source/data/data_shadeCalculate/{aim_location}/f_Directory_{aim_location}.csv')

pvg_epw_dataset.set_index('Hoy', inplace=True)
pvg_f_directory.set_index('Hoy', inplace=True)

# 计算的hoy列表
pvg_hoy = pd.read_csv(f'./source/data/hoys/hoy_{aim_location}/annual_hoy.txt', header=None)

pvg_log = pd.DataFrame(columns=[
    'hoy', 'sd_angle', 'beta', 'azimuth', 'zenith',
    'theta', 'epw_direct(W/m2)', 'surface_direct(W/m2)', 'epw_diffuse(W/m2)',
    'surface_diffuse(W/m2)', 'surface_reflect(W/m2)', 'surface_total_radiant']
)


def angle_between_vectors(v1, v2):  # v1: 太阳向量（计算中取反向向量）| v2: 表面法向量
    # 如果 v1 或 v2 是字符串，解析为数值列表
    if isinstance(v1, str):
        v1 = ast.literal_eval(v1)
    if isinstance(v2, str):
        v2 = ast.literal_eval(v2)

    # 转换为 NumPy 数组，并确保数据类型是浮动类型
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # 计算 v1 的反向向量
    v1 = -v1

    # 计算点积和模长
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 防止除以零的错误，确保向量的模长不为零
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("One of the vectors has zero magnitude, cannot compute angle.")

    # 计算余弦值并限制在合法的范围内
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # 防止浮点误差，确保 cos_theta 的值在 -1 到 1 之间
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算夹角的弧度值
    angle_rad = np.arccos(cos_theta)

    return angle_rad


class pvgCalculator:
    @staticmethod
    def calculateRadiant(shading_angle, hoy):
        """
        计算光伏板表面辐照量列表(Wh/m²)

        Args:
        - surface_azimuth (float): 表面倾角，正南则为0，南偏东为负，南偏西为正
        - shading_angle (float): 遮阳角度
        - hoy (int): 当前时间，用于取出太阳方位角、高度角、辐照和相关参数

        Returns:
        - Rad_direct_surface : 直射辐射
        - Rad_diffuse_surface : 漫射辐射
        - Rad_reflected : 反射辐射
        - total_radiant : 总辐照量
        """

        # 确保 hoy 是有效的索引
        if hoy not in pvg_epw_dataset.index:
            print(f"Warning: Hoy {hoy} is out of bounds. Skipping.")
            return None

        # 引用与取值（通过 .loc 使用 Hoy 作为索引）
        # 通过 epw 输入获取【太阳高度角 & 方位角】
        # azimuth = float(pvg_epw_dataset.loc[hoy, 'Azimuth'])  # 使用 loc 获取值
        # altitude = float(pvg_epw_dataset.loc[hoy, 'Altitude'])  # 使用 loc 获取值
        # zenith = math.radians(90 - altitude)
        # 根据佩雷兹天空模型计算得到的【太阳高度角 & 方位角】
        azimuth = math.radians(float(pvg_f_directory.loc[hoy, 'calculated_azimuth']))
        zenith = math.radians(float(pvg_f_directory.loc[hoy, 'calculated_zenith']))

        f1 = float(pvg_f_directory.loc[hoy, 'f1'])  # 使用 loc 获取值
        f2 = float(pvg_f_directory.loc[hoy, 'f2'])  # 使用 loc 获取值

        rad_direct = float(pvg_epw_dataset.loc[hoy, 'Direct_Rad'])  # 使用 loc 获取值
        rad_diffuse = float(pvg_epw_dataset.loc[hoy, 'Diffuse_Rad'])  # 使用 loc 获取值

        # 天顶角，即高度角的补角

        zenith_sin, zenith_cos = math.sin(zenith), math.cos(zenith)

        # beta，水平面与光伏表面夹角
        beta = - math.radians(90 - shading_angle)
        beta_deg = math.degrees(beta)
        beta_sin, beta_cos = math.sin(beta), math.cos(beta)

        # 使用 loc 获取太阳向量的字符串表示
        sun_vector_str = pvg_epw_dataset.loc[hoy, 'Sun_Vector']
        # 将字符串解析为列表
        sun_vector = ast.literal_eval(sun_vector_str)

        # 计算太阳向量与平面法向量夹角，我的计算方法 - 利用计算太阳向量和面法向量夹角
        sda_cos = math.radians(180 - shading_angle)
        sda = math.radians(shading_angle)
        surface_normal = [0, math.cos(sda_cos), math.sin(sda)]
        theta = angle_between_vectors(sun_vector, surface_normal)
        theta_cos = math.cos(theta)
        theta_deg = math.degrees(theta)

        # luang算法，公式计算
        # theta_cos = zenith_cos * beta_cos + zenith_sin * beta_sin * math.cos(azimuth)
        # theta = math.acos(theta_cos)
        # theta_deg = math.degrees(theta)

        # 计算 直射 | 漫射 辐射量
        rad_direct_surface = rad_direct * theta_cos  # 考虑入射角效应

        diffuse_amd = ((1 - f1) * ((1 + beta_cos) / 2)
                       + (f1 * max(0, theta_cos) / max(math.cos(math.radians(85)), zenith_cos))
                       + (f2 * beta_sin))

        # 调试输出
        # print('hoy', hoy)
        # print('f1', f1)
        # print('f2', f2)
        # print('theta_cos', theta_cos)
        # print('beta_sin', beta_sin)
        # print('diffuse_amd:', diffuse_amd)
        # print('diffuse_amd1', (1 - f1) * ((1 + beta_cos) / 2))
        # print('diffuse_amd2', f1 * max(0, zenith_cos) / max(math.cos(math.radians(85)), zenith_cos))
        # print('diffuse_amd3', f2 * beta_sin)
        # print('------------------------------')
        rad_diffuse_surface = rad_diffuse * diffuse_amd

        if rad_diffuse_surface < 0:
            rad_diffuse_surface = 0

        # 计算地面反射辐射量
        albedo = 0.2  # 地面反照率
        rad_reflected = (rad_direct + rad_diffuse) * albedo * (1 - beta_cos) / 2

        # 计算光伏表面总辐照量
        total_radiant = rad_direct_surface + rad_diffuse_surface + rad_reflected

        # 加入日志:记录计算过程数据
        log_entry = {
            'hoy': hoy,
            'sd_angle': shading_angle,
            'beta': beta_deg,
            'azimuth': azimuth,
            'zenith': zenith,
            'theta': theta_deg,
            'epw_direct(W/m2)': rad_direct,
            'surface_direct(W/m2)': rad_direct_surface,
            'epw_diffuse(W/m2)': rad_diffuse,
            'surface_diffuse(W/m2)': rad_diffuse_surface,
            'surface_reflect(W/m2)': rad_reflected,
            'surface_total_radiant': total_radiant
        }

        # 使用 concat 而不是 _append 来添加日志
        global pvg_log

        # 在执行 concat 前检查 pvg_log 是否为空：否则会报错
        log_entry_df = pd.DataFrame([log_entry])
        if not pvg_log.empty:
            pvg_log = pd.concat([pvg_log, log_entry_df], ignore_index=True)
        else:
            pvg_log = log_entry_df
        return total_radiant

    @staticmethod
    def calculateMaxRadiantList(hoy_list, shading_angles):
        """
        计算每小时光伏板表面最大辐照量(Wh/m²)

        Args:
        - hoy_list (list[int]): 需要计算的 hoy 列表
        - surface_azimuth (float): 表面倾角（角度）
        - shading_angles (float): 遮阳角度（角度）
        - window_transmittance (float): 窗透射率
        - hoy (int): 当前时间，用于取出太阳方位角、高度角、辐照和相关参数

        Returns:
        - Rad_direct_surface : 直射辐射
        - Rad_diffuse_surface : 漫射辐射
        - Rad_reflected : 反射辐射
        - total_radiant : 总辐照量
        """
        # 确保 hoy_list 是一个列表
        if not isinstance(hoy_list, list):
            hoy_list = list(hoy_list)

        max_rad_hourly = {}
        for hoy in hoy_list:
            hoy = int(hoy)  # 确保 hoy 是整数类型
            print(f"Processing Hoy: {hoy}")  # 调试打印当前处理的 Hoy
            max_rad = float('-inf')
            for shading_angle in shading_angles:
                rad = pvgCalculator.calculateRadiant(shading_angle, hoy)
                if rad is not None and rad > max_rad:
                    max_rad = rad
            max_rad_hourly[hoy] = max_rad  # 将 hoy 转为整数作为字典的键

            # 实时打印每小时最大辐照量
            # print(f"Hoy: {hoy}, Max Radiant: {max_rad:.2f} Wh/m²")

        max_radiant = pd.DataFrame({
            'Hoy': list(max_rad_hourly.keys()),
            'Max_Radiant': list(max_rad_hourly.values())
        })

        return max_radiant

    @staticmethod
    def calculateHoyPVGeneration(hoy, surface_radiant, panel_area, P_stc):
        # 获取温度值，使用 .loc 按 Hoy 查找
        T_surround = float(pvg_epw_dataset.loc[hoy, 'db_temperature'])

        # 计算光伏组件的实际输出功率
        delta_G = surface_radiant / 1000  # 辐照修正值 | G-stc一般为1000

        T_cell = T_surround + surface_radiant / 800 * (25 - 20)  # 光伏板温度：25是额定温度，20是常数
        delta_T = 1 - 0.0043 * (T_cell - 25)  # 温度修正值

        delta_P = delta_T * delta_G  # 功率修正值
        actual_power_output = P_stc * delta_P

        # 计算发电量
        pv_generation = actual_power_output * panel_area
        return pv_generation

    @staticmethod
    def calculatePvGeneration(hoys, surface_radiant_list, panel_area, P_stc):
        """
        计算光伏板发电量(Wh/m²)

        Args:
        - hoys (list): HOY 列表
        - surface_radiant_list (list): 表面辐照量列表
        - panel_area (float): 光伏板面积
        - conversion_efficiency (float): 转换效率

        Returns:
        - pv_generations_list : 包含 HOY 和 PV Generation 列的 Polars DataFrame
        """
        # 确保 hoys 和 surface_radiant_list 长度一致
        assert len(hoys) == len(surface_radiant_list), "Hoys 和表面辐照量列表长度不匹配"

        # 计算每个表面辐照量对应的光伏发电量
        pv_generations = []
        for i in range(len(hoys)):
            hoy = hoys[i]
            radiant = surface_radiant_list[i]

            # 确保 hoy 是标量，而非 Series
            if isinstance(hoy, pd.Series):
                hoy = hoy.iloc[0]  # 如果是 Series，取第一个元素

            # 确保 hoy 是有效的索引
            if hoy not in pvg_epw_dataset.index:
                print(f"Warning: Hoy {hoy} is out of bounds. Skipping.")
                continue

            pv_generation = pvgCalculator.calculateHoyPVGeneration(hoy, radiant, panel_area, P_stc)
            pv_generations.append(pv_generation)

            # # 实时打印每小时发电量
            # print(
            #     f"Hoy: {hoy}, Radiant: {radiant:.2f} W/m², "
            #     f"T_surround: {T_surround:.2f} °C, T_cell: {T_cell:.2f} °C, "
            #     f"delta_G: {delta_G:.4f}, delta_T: {delta_T:.4f}, delta_P: {delta_P:.4f},"
            #     f"Actual Power Output: {actual_power_output:.2f} W")

        pv_generations_list = pd.DataFrame({
            'Hoy': hoys,
            'Max_Radiant(W/m2)': surface_radiant_list,
            'PV_Generation(Wh)': pv_generations
        })

        return pv_generations_list


def main():
    angle_array = list(range(0, 91, 1))  # 遍历所有角度下的发电量
    hoys = pvg_hoy.squeeze().tolist()  # 确保 HOY 是列表

    # 修改日志输出，添加更多信息
    print("Calculating radiant values...")
    max_radiant = pvgCalculator.calculateMaxRadiantList(hoys, angle_array)
    print("Calculating PV generation...")
    max_pv_generation = pvgCalculator.calculatePvGeneration(
        hoys,
        max_radiant['Max_Radiant'],
        total_panel_area,
        100
    )

    # 将最大发电量与原始数据合并（按 HOY 对齐），并包括 Max_Radiant 列
    updated_epw_data = pvg_epw_dataset.merge(
        max_pv_generation[['Hoy', 'Max_Radiant(W/m2)', 'PV_Generation(Wh)']],
        how='left',
        left_on='Hoy',
        right_on='Hoy'
    )

    # 保存更新后的数据到同一文件
    updated_epw_file_path = f'./source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}_withPVG.csv'
    updated_epw_data.to_csv(updated_epw_file_path, index=False)
    print(f"Updated EPW data saved to {updated_epw_file_path}")

    # 保存详细日志
    print("Saving calculation log...")
    pvg_log.to_csv('pvg_detailed_log-250103-Perez.csv', index=False)
    print(f"Log saved with {len(pvg_log)} entries")


if __name__ == '__main__':
    main()
