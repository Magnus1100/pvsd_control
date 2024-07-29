import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 vis_dataset.txt 的路径，相对于 my_package 目录
vis_dataset_path = os.path.join(current_dir, '..', 'source', 'data', 'vis_dataset_2821.txt')
# 规范化路径
vis_dataset_path = os.path.abspath(vis_dataset_path)
# 打印路径以供调试
print(f"vis_dataset_path: {vis_dataset_path}")
# 读取 vis_dataset.txt 文件内容
with open(vis_dataset_path, 'r') as file:
    vis_data = file.read()

# 打印内容以供调试
print(vis_data)
