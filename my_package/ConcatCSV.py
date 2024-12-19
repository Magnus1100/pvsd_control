import os
import pandas as pd


def merge_csv_files(input_folder, output_file, sort_column):
    """
    汇总指定文件夹中的所有CSV文件，合并成一个文件。

    :param input_folder: 包含CSV文件的文件夹路径
    :param output_file: 输出的合并CSV文件路径
    """
    # 获取所有CSV文件的完整路径
    csv_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.csv')]

    # 检查是否有CSV文件
    if not csv_files:
        print("指定文件夹中没有CSV文件。")
        return

    # 存储读取的CSV文件
    data_frames = []

    # 逐个读取CSV文件
    for file in csv_files:
        try:
            df = pd.read_csv(file)  # 读取CSV文件
            data_frames.append(df)  # 加入列表
            print(f"已读取文件：{file}")
        except Exception as e:
            print(f"读取文件失败：{file}，错误信息：{e}")

    # 合并所有DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # 按指定列进行排序
    if sort_column in combined_df.columns:
        combined_df = combined_df.sort_values(by=sort_column, ascending=True)
        print(f"已按照 '{sort_column}' 列排序。")
    else:
        print(f"警告：列 '{sort_column}' 不存在，无法排序。")

    # 将合并后的数据保存到指定文件
    combined_df.to_csv(output_file, index=False)
    print(f"所有文件已合并，输出文件路径：{output_file}")


if __name__ == "__main__":
    # 输入文件夹路径和输出文件路径
    main_input_folder = r'./source/pvsd_schedule/bj-241217'  # 替换为CSV文件所在的目录路径
    main_output_file = r'shading_schedule/bj-241217/pvsd_schedule.csv'  # 输出的文件名和路径
    main_sort_column = "Hoy"  # 指定要排序的列名

    # 调用合并函数
    merge_csv_files(main_input_folder, main_output_file, main_sort_column)
