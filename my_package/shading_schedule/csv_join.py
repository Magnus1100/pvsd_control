import os
import pandas as pd

# 设置文件夹路径
folder_path = "bj-1100-250114"
output_file = "bj-1100-250114/bj_1100_schedule-250117.csv"

# 获取文件夹中所有的CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个空的DataFrame，用于存放整合后的数据
merged_df = pd.DataFrame()

# 遍历每个CSV文件并合并
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 合并到总的DataFrame中
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# 将合并后的DataFrame保存为一个新的CSV文件
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"整合完成，文件已保存为 {output_file}")
