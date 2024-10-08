import matplotlib.pyplot as plt

# 读取数据
with open('vis_8.txt', 'r') as file:
    # 假设每行一个数值
    data = [float(line.strip()) / 100 for line in file if line.strip()]

# 计算比例
thresholds = [0.4, 0.5, 0.6, 0.7]
proportions = {threshold: sum(value > threshold for value in data) / len(data) for threshold in thresholds}

# 输出比例
for threshold, proportion in proportions.items():
    print(f"高于{threshold}的比例: {proportion:.2%}")

# 绘制数据图
plt.figure(figsize=(10, 5))
plt.hist(data, bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0.4, color='red', linestyle='--', label='Threshold 0.4')
plt.axvline(x=0.5, color='orange', linestyle='--', label='Threshold 0.5')
plt.axvline(x=0.6, color='green', linestyle='--', label='Threshold 0.6')
plt.axvline(x=0.7, color='purple', linestyle='--', label='Threshold 0.7')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('数据占比')
plt.legend()
plt.grid()
plt.show()