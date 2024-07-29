from eppy.modeleditor import IDF
from pyfmi import load_fmu

fmu_file_path = r'D:\pythonProject\pythonProject\idf_test.fmu'
fmu_model = load_fmu('your_model.fmu')

# 初始化模型
fmu_model.setup_experiment(start_time=0.0)

# 设置模拟参数
start_hoy = 0.0
end_hoy = 10.0
time_step = 1.0

# idf路径
idf_template_path = r'C:\EnergyPlusV24-1-0\Energy+.idd'

# 设置EnergyPlus的idf文件模板路径
IDF.setiddname(idf_template_path)
idf_file_path = (r'D:\pythonProject\pythonProject\.venv\my_package\source\rhino_energy_model\NoShadeModel\openstudio'
                 r'\run\in.idf')

# 逐时模拟
current_time = start_hoy
while current_time <= end_hoy:
    # 执行一步模拟
    fmu_model.do_step(current_time, time_step, True)

    # 获取 ExternalInterface 变量
    shade_angle = fmu_model.get('shade_angle')  # 替换为你的变量名
    print(f'Time: {current_time:.2f} hours, Shade Angle: {shade_angle:.2f}')

    # 更新当前时间
    current_time += time_step

# 读取IDF文件
idf = IDF(idf_file_path)

PvShadeBlind1 = idf.newidfobject(
    "Shading:Overhang",
    Name="PvShadeBlind1",
    Window_or_Door_Name="Window_3732ef8f",
    Height_above_Window_or_Door=0,
    Tilt_Angle_from_Window_or_Door=0,
    Left_extension_from_Window_or_Door_Width=0,
    Right_extension_from_Window_or_Door_Width=0,  # left + right + window_width = 遮阳宽度
    Depth=0.5
)
print(PvShadeBlind1)

idf_FMU = idf.newidfobject(
    "ExternalInterface:FunctionalMockupUnitImport",
    FMU_File_Name='myFmu',
    FMU_Timeout=0,
    FMU_LoggingOn=0
)
print(idf_FMU)

fmu_var = idf.newidfobject(
    "ExternalInterface:FunctionalMockupUnitImport:To:Variable",
    Name="shade_angle",  # 模型中使用的变量名称（要传递到 FMU 的变量名）
    FMU_File_Name='myFmu',  # 要使用的 FMU 文件名
    FMU_Instance_Name="sd_angle",  # 标识特定的 FMU 实例
    FMU_Variable_Name="sd_angle",  # 在 FMU 中更新的变量名称（表示在 FMU 内部需要接收数据的变量）
    Initial_Value=0  # 变量的初始值,在模拟开始时将该值发送给 FMU
)
print(fmu_var)

"""
在 ExternalInterface:FunctionalMockupUnitImport 的上下文中，To:Variable 和 To:Schedule 的区别主要在于数据的类型和用途：
To:Variable 表示将数据作为变量传递给 FMU。通常针对特定的状态或参数，如温度、角度等。这些变量可以在 FMU 中用于实时计算和模拟。
To:Schedule 表示将数据作为调度信息传递给 FMU。这通常涉及时间序列数据或周期性事件，用于控制模型的行为。调度数据可以是预定义的时间表，决定何时执行特定操作或使用特定参数。

总结来说，To:Variable 关注的是具体的输入参数，而 To:Schedule 则关注于时间和频率的控制。
"""

# output_path = 'idf_test0705_V1.idf'
#
# # 保存修改后的IDF文件
# idf.save(output_path)

#
# idf_EMS_Sensor = idf.newidfobject(
#     "EnergyManagementSystem:Sensor",
#     Name="ShadePositionSensor",
# )
#
# idf_EMS_Actuator = idf.newidfobject(
#     "EnergyManagementSystem:Actuator",
#     Name="ShadeControlActuator",
#     Actuated_Component_Unique_Name="MyWindow",
#     Actuated_Component_Type="WindowShadeControl",
#     Actuated_Component_Control_Type="ControlType",
# )
#
# idf_EMS_Program = idf.newidfobject(
#     "EnergyManagementSystem:Subroutine",
#     Name="ControlShadingSubroutine",
#     Program_Line_1="IF (ShadePositionSensor < 0.5)",
#     Program_Line_2="SET ShadeControlActuator = 1",
#     Program_Line_3="ELSE",
#     Program_Line_4="SET ShadeControlActuator = 0",
#     Program_Line_5="ENDIF",
# )
#
# idf_EMS = idf.newidfobject(
#     "EnergyManagementSystem:ProgramCallingManager",
#     Name="ControlShadingCallingManager",
#     EnergyPlus_Model_Calling_Point="BeginTimestepBeforePredictor",
#     Program_Name_1="ControlShadingProgram",
#
# )
#
# # 创建遮阳参数Schedule
# schedule_name = "ShadingSchedule"
# schedule = idf.newidfobject(
#     "Schedule:File",
#     Name="ShadingSchedule",
#     Schedule_Type_Limits_Name="Fraction",
#     File_Name="shading_data.csv",
#     Column_Number=2,
#     Rows_to_Skip_at_Top=1,
#     Number_of_Hours_of_Data=8760,
#     Column_Separator="Comma"
# )
#
# # 创建遮阳控制程序
# subroutine_name = "ControlShadingSubroutine"
# subroutine = idf.newidfobject(
#     "EnergyManagementSystem:Subroutine",
#     Name=subroutine_name,
#     Program_Line_1="SET ShadeControlActuator = @TrendValue " + schedule_name + " 1"
# )
#
# # 创建执行器
# EMS_Actuator = idf.newidfobject(
#     "EnergyManagementSystem:Actuator",
#     Name="ShadeControlActuator",
#     Actuated_Component_Unique_Name="MyWindow",
#     Actuated_Component_Type="Window Shading Control",
#     Actuated_Component_Control_Type="Control Type"
# )
#
# # 创建程序
# program_name = "ControlShadingProgram"
# EMS_Program = idf.newidfobject(
#     "EnergyManagementSystem:Program",
#     Name=program_name,
#     Program_Line_1="RUN " + subroutine_name
# )
#
# # 创建程序调用管理器
# EMS_ProgramCallingManager = idf.newidfobject(
#     "EnergyManagementSystem:ProgramCallingManager",
#     Name="ControlShadingCallingManager",
#     EnergyPlus_Model_Calling_Point="BeginTimestepBeforePredictor",
#     Program_Name_1=program_name
# )
#
# # 创建传感器
# EMS_Sensor = idf.newidfobject(
#     "EnergyManagementSystem:Sensor",
#     Name="ShadePositionSensor",
#     # Output_Variable_or_Output_Meter_Index_Key_Name="MyWindow",
#     # Output_Variable_or_Output_Meter_Name="Window Shade Position"
# )
#
# '''
# 在idf中添加外部参数
# '''
# # 添加 ExternalInterface 对象
# EI_key = "ExternalInterface"
# EI_idf = idf.newidfobject(
#     EI_key,
#     Name="myEI"
# )
#
# # 添加 ExternalInterface:FunctionalMockupUnitExport:To Schedule对象
# EI_fmu_ts_key = "ExternalInterface:FunctionalMockupUnitExport:To:Schedule"
# EI_fmu_ts_idf = idf.newidfobject(
#     EI_fmu_ts_key,
#     Name="myEI:ToSchedule",
#     Schedule_Name="myEI_schedule",
#     Initial_Value=0,
# )
#
# # 遮阳角度参数
# idf_sd_angle = idf.newidfobject(  # shade_angle
#     "ExternalInterface:Variable",
#     Name="ShadeAngle",
#     Initial_Value=90,
# )
# # 遮阳位置参数
# idf_sd_location = idf.newidfobject(  # shade location
#     "ExternalInterface:Variable",
#     Name="ShadeLocation",
#     Initial_Value=0,
# )


# PvShadeBlind2 = idf.newidfobject(
#     "Shading:Zone:Detailed",
#     Name="PvShadeBlind2",
#     Base_Surface_Name="Window_1030049a",
#     Transmittance_Schedule_Name="ShadingSchedule",
#     Number_of_Vertices=8,
#     Vertex_2_Xcoordinate="[0, 0, 0]",
# )

# print(schedule, subroutine, EMS_Actuator, EMS_Program, EMS_ProgramCallingManager, EMS_Sensor)
