import pandas as pd
import matplotlib.pyplot as plt

# 定义输出文件的路径
output_file_path = r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_result\epluszsz.csv'

# 读取 CSV 文件
try:
    df = pd.read_csv(output_file_path)
    print("文件读取成功！")
except FileNotFoundError:
    print(f"文件 {output_file_path} 未找到。请检查文件路径。")
    exit()

# 显示数据框的前几行
print(df.head())

# 假设逐时能耗数据在特定列中（例如 'Electricity:Facility [J](Hourly)'）
# 你可以根据实际的列名进行调整
if 'Electricity:Facility [J](Hourly)' in df.columns:
    hourly_energy = df['Electricity:Facility [J](Hourly)']
    print(f"逐时能耗数据：\n{hourly_energy}")
else:
    print("列 'Electricity:Facility [J](Hourly)' 未在 CSV 文件中找到。请检查列名。")

# 检查是否有逐时能耗数据列
if 'Electricity:Facility [J](Hourly)' in df.columns:
    hourly_energy = df['Electricity:Facility [J](Hourly)']

    # 绘制逐时能耗数据的图表
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_energy, label='Electricity:Facility [J](Hourly)')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Energy (J)')
    plt.title('Hourly Energy Consumption')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("列 'Electricity:Facility [J](Hourly)' 未在 CSV 文件中找到。请检查列名。")
