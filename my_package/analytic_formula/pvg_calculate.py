import math
import polars as pl
from my_package.analytic_formula import epw_data_file as edf

#  全局变量
pv_use_ratio = 0.6


class pvgCalculator:  # 计算光伏发电量
    @staticmethod
    def calculateIrradiance(surface_azimuth, shading_angle, window_transmittance, hoy):
        """
        计算光伏板表面辐照量(Wh/m²)

        Args:
        - surface_azimuth (float): 表面倾角，正南则为0，南偏东为负，南偏西为正
        - shading_angle (float): 遮阳角度
        - window_transmittance (float): 窗透射率
        - hoy (int): 当前时间，用于取出太阳方位角、高度角、辐照和相关参数

        Returns:
        - I_direct_surface : 直射辐射
        - I_diffuse_surface : 漫射辐射
        - I_reflected : 反射辐射
        - total_irradiance : 总辐照量
        """
        # 引用与取值
        solar_data = edf.epw_dataset
        altitude = solar_data['Altitude'][hoy]
        azimuth = solar_data['Azimuth'][hoy]
        f1 = edf.f_directory['f1'][hoy]
        f2 = edf.f_directory['f2'][hoy]
        I_direct = solar_data['I_direct'][hoy]
        I_diffuse = solar_data['I_diffuse'][hoy]

        theta_cos = altitude * math.cos(azimuth)
        beta_cos = 90 - shading_angle

        # 将角度转换为弧度
        solar_elevation_rad = math.radians(solar_data["Altitude"][hoy])
        solar_azimuth_rad = math.radians(solar_data["Azimuth"][hoy])
        surface_azimuth_rad = math.radians(surface_azimuth)
        shading_angle_rad = math.radians(shading_angle)
        beta_cos_rad = math.radians(beta_cos)

        # 计算太阳入射角余弦值 (cos(θ))
        cos_theta = (math.sin(solar_elevation_rad) * math.cos(shading_angle_rad) +
                     math.cos(solar_elevation_rad) * math.sin(shading_angle_rad) *
                     math.cos(solar_azimuth_rad - surface_azimuth_rad))
        # 如果 cos_theta 小于0，则表明没有直射辐射
        if cos_theta < 0:
            cos_theta = 0

        # 计算直射/漫射辐射量
        I_direct_surface = I_direct * cos_theta
        I_diffuse_surface = I_direct * (
                (1 - f1) * ((1 + beta_cos) / 2) +
                (f1 * max(0, theta_cos) / max(math.cos(1.48353), f1)) +
                (f2 * beta_cos_rad)
        )

        # 计算地面反射辐射量
        albedo = 0.2
        I_reflected = albedo * I_diffuse

        # 计算窗内总辐照量
        total_irradiance = (I_direct_surface + I_diffuse_surface + I_reflected) * window_transmittance

        return total_irradiance

    @staticmethod
    def calculateMaxIrradianceList(hoy_list, surface_azimuth, shading_angles, window_transmittance):
        """
        计算每小时光伏板表面最大辐照量(Wh/m²)

        Args:
        - hoy_list (list[int]): 需要计算的 hoy 列表
        - surface_azimuth (float): 表面倾角（角度）
        - shading_angles (float): 遮阳角度（角度）
        - window_transmittance (float): 窗透射率
        - hoy (int): 当前时间，用于取出太阳方位角、高度角、辐照和相关参数

        Returns:
        - I_direct_surface : 直射辐射
        - I_diffuse_surface : 漫射辐射
        - I_reflected : 反射辐射
        - total_irradiance : 总辐照量
        """
        max_rad_hourly = {}  # 用字典记录每个小时的最大辐照量
        for hoy in hoy_list:
            max_rad = float('-inf')  # 初始化每小时的最大辐照量为负无穷大
            for shading_angle in shading_angles:
                rad = pvgCalculator.calculateIrradiance(surface_azimuth, shading_angle, window_transmittance, hoy)
                if rad is not None and rad > max_rad:
                    max_rad = rad
            max_rad_hourly[hoy] = max_rad

        # 转换为 Polars DataFrame
        max_radiance = pl.DataFrame({
            'HOY': list(max_rad_hourly.keys()),
            'Max_Irradiance': list(max_rad_hourly.values())
        })

        return max_radiance

    @staticmethod
    def calculatePvGeneration(hoys, surface_irradiances, panel_area, conversion_efficiency):
        """
        计算光伏板发电量(Wh/m²)

        Args:
        - hoys (list): HOY 列表
        - surface_irradiances (list): 表面辐照量列表
        - panel_area (float): 光伏板面积
        - conversion_efficiency (float): 转换效率

        Returns:
        - pv_generations_list : 包含 HOY 和 PV Generation 列的 Polars DataFrame
        """
        # 确保 hoys 和 surface_irradiances 长度一致
        assert len(hoys) == len(surface_irradiances), "HOYs 和表面辐照量列表长度不匹配"

        # 计算每个表面辐照量对应的光伏发电量
        pv_generations = []
        for irradiance in surface_irradiances:
            # 计算光伏组件接收到的总辐照量
            total_irradiance = irradiance * panel_area

            # 计算光伏组件的实际输出功率
            actual_power_output = total_irradiance * pv_use_ratio  # 假设光伏组件的利用率为60%

            # 计算发电量
            pv_generation = actual_power_output * conversion_efficiency
            pv_generations.append(pv_generation)

        # 将结果转换为 Polars 的 DataFrame
        pv_generations_list = pl.DataFrame({
            'HOY': hoys,
            'Max_Irradiance': surface_irradiances,
            'PV_Generation(Wh)': pv_generations
        })

        return pv_generations_list

    @staticmethod
    def calculateHoyPvGeneration(radiance, panel_area, conversion_efficiency):
        """
        计算每个小时具体的发电量
        """
        return radiance * panel_area * conversion_efficiency * pv_use_ratio  # 假设光伏组件的利用率为60%


def main():
    angle_array = list(range(0, 91, 1))
    max_irradiance = pvgCalculator.calculateMaxIrradianceList(edf.hoys, 0, angle_array, 0.6)
    max_pv_generation = pvgCalculator.calculatePvGeneration(edf.hoys, max_irradiance['Max_Irradiance'], 16, 0.5)

    edf.epw_dataset['max_irradiance'] = max_pv_generation['Max_Irradiance']
    edf.epw_dataset['max_pv_generation'] = max_pv_generation['PV_Generation(Wh)']
    edf.epw_dataset.to_csv('epw_data.csv')
    # print(edf.epw_dataset)


if __name__ == '__main__':
    main()
