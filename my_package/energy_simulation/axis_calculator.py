import math

class CalculateBlindAxis:
    @staticmethod
    def CalculatePoints(p1, p2, shade_length, shade_angle, shade_position, blind_count):
        shade_angle = math.radians(shade_angle)
        shade_interval = abs(shade_position)
        # print("shade_interval:"+str(shade_interval))
        y_changed = shade_length * math.sin(shade_angle)
        z_changed = shade_length * math.cos(shade_angle)
        print(y_changed, z_changed)

        if shade_position <= 0:
            point1 = [p1[0], p1[1], p1[2]]
            point2 = [p1[0], p1[1] + y_changed, p1[2] + z_changed]
            point3 = [p2[0], p1[1] + y_changed, p2[2] + z_changed]
            point4 = [p2[0], p1[1], p1[2]]
        else:
            print(p1[2],shade_interval * (blind_count - 1))
            point1 = [p1[0], p1[1], p1[2] - shade_interval * (blind_count - 1)]
            point2 = [p1[0], p1[1] + y_changed, p1[2] + z_changed - shade_interval * (blind_count - 1)]
            point3 = [p2[0], p1[1] + y_changed, p2[2] + z_changed - shade_interval * (blind_count - 1)]
            point4 = [p2[0], p1[1], p1[2] - shade_interval * (blind_count - 1)]

        point_list = [point1, point2, point3, point4]
        return point_list

    @staticmethod
    def GetBlindAxis(point_list, blind_count, shade_position):
        blind_axis = []
        shade_interval = 0.15 - abs(shade_position)
        print(shade_interval)

        for i in range(blind_count):
            adjusted_points = []
            for point in point_list:
                new_point = [point[0], point[1], point[2] - shade_interval * i]
                adjusted_points.append(new_point)
            blind_axis.append(adjusted_points)
        return blind_axis
