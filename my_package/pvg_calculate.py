import math
import polars as pl
import pandas as pd

#  全局变量
# pv_use_ratio = 0.6
aim_location = 'bj'
single_panel_area, total_panel_area = 0.306, 4.896
pvg_epw_dataset = pd.read_csv(f'./source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}.csv')
pvg_f_directory = pd.read_csv(f'./source/data/data_shadeCalculate/{aim_location}/f_Directory_{aim_location}.csv')

pvg_epw_dataset.set_index('Hoy', inplace=True)
pvg_f_directory.set_index('Hoy', inplace=True)

pvg_hoy = pd.read_csv(f'./source/data/hoys/annual_hoy.txt', header=None)
pvg_hoy = [8, 9, 10]


class pvgCalculator:  # 计算光伏发电量
    @staticmethod
    def calculateRadiant(surface_azimuth, shading_angle, hoy):
        """
        计算光伏板表面辐照量(Wh/m²)

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

        # 引用与取值
        azimuth = pvg_epw_dataset['Azimuth'][hoy]
        altitude = pvg_epw_dataset['Altitude'][hoy]
        f1 = pvg_f_directory['f1'][hoy]
        f2 = pvg_f_directory['f2'][hoy]
        rad_direct = pvg_epw_dataset['Direct_Rad'][hoy]
        rad_diffuse = pvg_epw_dataset['Diffuse_Rad'][hoy]

        theta_cos = altitude * math.cos(azimuth)
        beta_cos = 90 - shading_angle
        beta = math.acos(math.radians(beta_cos))
        beta_sin = math.sin(beta)

        # 将角度转换为弧度
        solar_azimuth_rad = math.radians(azimuth)
        solar_elevation_rad = math.radians(altitude)
        surface_azimuth_rad = math.radians(surface_azimuth)
        shading_angle_rad = math.radians(shading_angle)

        beta_cos_rad = math.radians(beta_cos)
        theta_cos_rad = math.radians(theta_cos)

        # 方位角余弦值
        zenith_cos = math.cos(azimuth)

        # 计算太阳入射角余弦值 (cos(θ))
        cos_theta = (math.sin(solar_elevation_rad) * math.cos(shading_angle_rad) +
                     math.cos(solar_elevation_rad) * math.sin(shading_angle_rad) *
                     math.cos(solar_azimuth_rad - surface_azimuth_rad))
        # 如果 cos_theta 小于0，则表明没有直射辐射
        if cos_theta < 0:
            cos_theta = 0

        # 计算 直射|漫射 辐射量
        rad_direct_surface = rad_direct * cos_theta  # 考虑入射角效应
        diffuse_amd = ((1 - f1) * ((1 + beta_cos_rad) / 2) +
                       (f1 * max(0, theta_cos_rad) / max(math.cos(1.48353), zenith_cos)) +
                       (f2 * beta_sin))
        rad_diffuse_surface = rad_direct * diffuse_amd

        # 计算地面反射辐射量
        albedo = 0.2
        rad_reflected = albedo * rad_diffuse

        # 计算光伏表面总辐照量
        total_radiant = rad_direct_surface + rad_diffuse_surface + rad_reflected

        print('diffuse_amd', diffuse_amd)
        print('beta_cos_rad: ', beta_cos_rad)
        print('zenith_cos: ', zenith_cos)
        print('theta_cos: ', theta_cos)
        print('rad_diffuse_surface: ', rad_diffuse_surface)
        print('total_radiant: ', total_radiant)

        return total_radiant

    @staticmethod
    def calculateMaxRadiantList(hoy_list, surface_azimuth, shading_angles):
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
            print(f"Processing Hoy: {hoy}")  # 调试打印当前处理的 Hoy
            max_rad = float('-inf')
            for shading_angle in shading_angles:
                rad = pvgCalculator.calculateRadiant(surface_azimuth, shading_angle, hoy)
                if rad is not None and rad > max_rad:
                    max_rad = rad
            max_rad_hourly[hoy] = max_rad

        max_radiant = pl.DataFrame({
            'Hoy': list(max_rad_hourly.keys()),
            'Max_Radiant': list(max_rad_hourly.values())
        })

        return max_radiant

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
            T_surround = pvg_epw_dataset['db_temperature'][hoy]
            # 计算光伏组件的实际输出功率
            delta_G = radiant / 1000  # 辐照修正值 | G-stc一般为1000
            T_cell = T_surround + radiant * (25 - 20)  # 公式：25是额定温度，20是常数
            delta_T = 1 + -0.0043 * (T_cell - 25)  # 温度修正值
            actual_power_output = P_stc * delta_T * delta_G

            # 计算发电量
            pv_generation = actual_power_output * panel_area
            pv_generations.append(pv_generation)

        pv_generations_list = pl.DataFrame({
            'Hoy': hoys,
            'Max_Radiant': surface_radiant_list,
            'PV_Generation(Wh)': pv_generations
        })

        return pv_generations_list


def main():
    angle_array = list(range(0, 91, 1))
    # hoy_list = pvg_hoy.squeeze().tolist()  # 确保 HOY 是列表
    max_radiant = pvgCalculator.calculateMaxRadiantList(pvg_hoy, 0, angle_array)
    max_pv_generation = pvgCalculator.calculatePvGeneration(
        pvg_hoy,
        max_radiant['Max_Radiant'],
        total_panel_area,
        0.5
    )

    # pvg_epw_dataset['max_Radiant'] = max_pv_generation['Max_Radiant']
    # pvg_epw_dataset['max_pv_generation'] = max_pv_generation['PV_Generation(Wh)']
    # pvg_epw_dataset.to_csv(f'epwData_{aim_location}.csv')
    # print(pvg_epw_dataset.shape)


if __name__ == '__main__':
    main()
