import polars as pl
import numpy as np
from eppy.modeleditor import IDF
from energy_simulation import energyplus_run as er

# 设置IDD文件路径
idf_template_path = r'C:\EnergyPlusV24-1-0\Energy+.idd'
IDF.setiddname(idf_template_path)

# 创建IDF对象
idf_file_path = r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_model\test__\openstudio\run\in.idf'
idf = IDF(idf_file_path)

# 生成逐时遮阳位置数据
hours = np.arange(1, 8761)
shading_positions = np.random.uniform(0, 0.15, size=8760)
position_data = pl.DataFrame({"Hour": hours, "ShadingPosition": shading_positions})

# 生成逐时遮阳角度数据
shading_angles = np.random.uniform(0, 90, size=8760)
angle_data = pl.DataFrame({"Hour": hours, "ShadingAngle": shading_angles})

# 保存生成的数据到CSV文件（EnergyPlus需要读取CSV文件）
position_data.write_csv('shading_position_data.csv')
angle_data.write_csv('shading_angle_data.csv')

# 创建遮阳位置Schedule
position_schedule_name = "ShadingPositionSchedule"
idf.newidfobject(
    "Schedule:File",
    Name=position_schedule_name,
    Schedule_Type_Limits_Name="Fraction",
    File_Name="shading_position_data.csv",
    Column_Number=2,
    Rows_to_Skip_at_Top=1,
    Number_of_Hours_of_Data=8760,
    Column_Separator="Comma"
)

# 创建遮阳角度Schedule
angle_schedule_name = "ShadingAngleSchedule"
idf.newidfobject(
    "Schedule:File",
    Name=angle_schedule_name,
    Schedule_Type_Limits_Name="Fraction",
    File_Name="shading_angle_data.csv",
    Column_Number=2,
    Rows_to_Skip_at_Top=1,
    Number_of_Hours_of_Data=8760,
    Column_Separator="Comma"
)

# 创建遮阳控制子例程
subroutine_name = "ControlShadingSubroutine"
idf.newidfobject(
    "EnergyManagementSystem:Subroutine",
    Name=subroutine_name,
    Program_Line_1="SET ShadeControlActuatorPosition = @TrendValue " + position_schedule_name + " 1",
    Program_Line_2="SET ShadeControlActuatorAngle = @TrendValue " + angle_schedule_name + " 1"
)

# 创建执行器用于遮阳位置
idf.newidfobject(
    "EnergyManagementSystem:Actuator",
    Name="ShadeControlActuatorPosition",
    Actuated_Component_Unique_Name="MyWindow",
    Actuated_Component_Type="Window Shading Control",
    Actuated_Component_Control_Type="Position"
)

# 创建执行器用于遮阳角度
idf.newidfobject(
    "EnergyManagementSystem:Actuator",
    Name="ShadeControlActuatorAngle",
    Actuated_Component_Unique_Name="MyWindow",
    Actuated_Component_Type="Window Shading Control",
    Actuated_Component_Control_Type="Angle"
)

# 创建程序
program_name = "ControlShadingProgram"
idf.newidfobject(
    "EnergyManagementSystem:Program",
    Name=program_name,
    Program_Line_1="RUN " + subroutine_name
)

# 创建程序调用管理器
idf.newidfobject(
    "EnergyManagementSystem:ProgramCallingManager",
    Name="ControlShadingCallingManager",
    EnergyPlus_Model_Calling_Point="BeginTimestepBeforePredictor",
    Program_Name_1=program_name
)

# 创建传感器用于遮阳位置
idf.newidfobject(
    "EnergyManagementSystem:Sensor",
    Name="ShadePositionSensor",
    # Output_Variable_or_Output_Meter_Index_Key_Name="MyWindow",
    # Output_Variable_or_Output_Meter_Name="Window Shade Position"
)

# 创建传感器用于遮阳角度
idf.newidfobject(
    "EnergyManagementSystem:Sensor",
    Name="ShadeAngleSensor",
    # Output_Variable_or_Output_Meter_Name="Window Shade Angle"
)

# 保存更新后的IDF文件
updated_idf_file = "path_to_save_updated_idf_file.idf"
idf.save(updated_idf_file)

# 运行EnergyPlus仿真
my_energyplus_exe = "energyplus"
my_epw_file = (r'D:\pythonProject\pythonProject\.venv\my_package\source\epw_file\CHN_Guangdong.Shenzhen'
               r'.594930_SWERA.epw')
my_idf_file = r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_model\test__\openstudio\run\in.idf'
my_output_dir = r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_result'

my_result = er.run_energyplus_simulation(my_energyplus_exe, my_epw_file, my_idf_file, my_output_dir)
print(my_result)
