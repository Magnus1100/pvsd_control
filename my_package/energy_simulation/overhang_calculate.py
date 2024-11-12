import pprint


def GetOverhangParameters(sd_position, sd_count, window_height):
    # 声明变量
    sd_interval = 0.15 - abs(sd_position)
    parameter_list = []
    final_slat_position = window_height - window_height / sd_count  # 最下部板的位置参数

    if sd_position <= 0:
        for i in range(sd_count):
            parameter_list.append(sd_interval * i)
    else:
        for i in range(sd_count):
            parameter_list.append(final_slat_position - sd_interval * i)
            parameter_list.sort()

    return parameter_list

