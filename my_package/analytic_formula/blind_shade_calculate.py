import math as mt
import pandas as pd

epw_data = pd.read_csv(r'D:\03-GitHub\pvsd_control\my_package\source\dataset\epw_data.csv')
vis_data_path = r'D:\03-GitHub\pvsd_control\my_package\source\dataset\vis_data_outside_0920.csv'
if 'Hoy' in epw_data.columns:
    # 将 'hoy' 列设置为索引
    epw_data = epw_data.set_index('Hoy')


class pvShadeBlind:
    def __init__(self, sd_width, sd_length, pv_panel_area, pv_efficiency, window_azimuth, window_transmittance,
                 slat_count, window_height):
        self.sd_width = sd_width
        self.sd_length = sd_length
        self.panel_area = pv_panel_area
        self.pv_efficiency = pv_efficiency
        self.window_azimuth = window_azimuth
        self.slat_count = slat_count
        self.window_height = window_height
        self.window_transmittance = window_transmittance


class ShadeCalculate:
    # 遮挡计算&视野获取
    @staticmethod
    def GetAngle(hoy, name):
        angle = epw_data.loc[int(hoy), name]
        radi = mt.radians(angle)
        return radi

    @staticmethod
    def GetVis(sd_angle, sd_location):
        vis_data = pd.read_csv(vis_data_path)
        vis = vis_data[(vis_data['sd_angle'] == sd_angle) &
                       (vis_data['sd_position'] == sd_location)]['vis'].values
        vis = vis / 100
        return vis

    @staticmethod
    def VerticalShadePercent(sd_width, sd_interval, ver_angle, sd_angle):
        angle1 = abs(sd_angle - ver_angle)
        x = sd_interval * mt.sin(sd_angle)
        y = x * mt.tan(angle1)
        z = sd_interval * mt.cos(sd_angle)

        if sd_angle > ver_angle:
            ver_unshade_area = z + y
        else:
            ver_unshade_area = z - y

        # 遮挡比率
        ver_sd_percent = 1 - (ver_unshade_area / sd_width)
        if ver_sd_percent < 0 or ver_angle > mt.pi:
            ver_sd_percent = 0
        if ver_sd_percent > 1:
            ver_sd_percent = 1

        return ver_sd_percent

    @staticmethod
    def HorizontalShadePercent(sd_length, sd_interval, hor_angle):

        hor_unshade_area = sd_interval / mt.tan(hor_angle)
        hor_sd_percent = 1 - (hor_unshade_area / sd_length)

        # 排除极端情况
        if hor_sd_percent < 0:
            hor_sd_percent = 0
        if hor_sd_percent > 1:
            hor_sd_percent = 1

        return hor_sd_percent

    @staticmethod
    def AllShadePercent(sd_length, sd_width, sd_interval, ver_angle, hor_angle, sd_angle):
        ver_percent = ShadeCalculate.VerticalShadePercent(sd_width, sd_interval, ver_angle, sd_angle)
        hor_percent = ShadeCalculate.HorizontalShadePercent(sd_length, sd_interval, hor_angle)
        return ver_percent * hor_percent
