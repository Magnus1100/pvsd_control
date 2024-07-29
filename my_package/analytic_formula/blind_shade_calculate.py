import math as mt
from my_package.analytic_formula import epw_data_file as ed
from my_package.analytic_formula import vis_data as vd


class pvShadeBlind:
    def __init__(self, sd_width, sd_length, pv_panel_area, pv_efficiency, window_azimuth, slat_count, window_height):
        self.sd_width = sd_width
        self.sd_length = sd_length
        self.panel_area = pv_panel_area
        self.pv_efficiency = pv_efficiency
        self.window_azimuth = window_azimuth
        self.slat_count = slat_count
        self.window_height = window_height


class ShadeCalculate:
    @staticmethod
    def GetAngle(hoy, name):
        angle = ed.epw_dataset.loc[int(hoy), name]
        radi = mt.radians(angle)
        return radi

    @staticmethod
    def GetVis(sd_angle, sd_location):
        vis = vd.vis_dataset[(vd.vis_dataset['sd_angle'] == sd_angle) &
                             (vd.vis_dataset['sd_location'] == sd_location)]['vis'].values
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
