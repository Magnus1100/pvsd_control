import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

"""
功能：评估随机森林模型的回归效果
使用步骤：
1.更改数据路径
2.更改模型路径
3.点击运行，更改test_size和random_state多试几次 -> 效果好的话就可以应用模型啦！
"""

fig_save_path = '../source/model_optimizer/model_bj1210/sDGP_NW_RF_1210.png'

# 指定特征与目标
df_normalized_path = r'../source/data/1126335/beijing/bj_241210_normalizedDataset.csv'

df_normalized = pd.read_csv(df_normalized_path)

# x = df_normalized[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
x = df_normalized[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval','Direct Radiation']]
y = df_normalized[['sDGP']]
z = df_normalized[['sUDI']]

# 划分测试集
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.00001, random_state=50)

# 加载模型
model_sdgp = joblib.load(r'../source/model_optimizer/model_bj1210/sDGP_NW_RF_1210bj.pkl')

# 使用模型预测测试集
y_pred = model_sdgp.predict(x_test)
# 确保 y_true 是一维的
y_true = y_true.values.flatten()

# 计算R方
r2 = r2_score(y_true, y_pred)
print(f"R-squared (R2 Score): {r2:.3f}")

# 计算均方误差（MSE）
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# 创建序号
indices = range(len(y_pred))

# 绘制预测值和真实值的关系图
plt.figure(figsize=(10, 10))  # 设置高分辨率的长图

# 绘制散点图
plt.scatter(y_true, y_pred, label='Predicted vs True', color='blue', s=10)

# 绘制理想曲线
max_val = max(max(y_true), max(y_pred))
min_val = min(min(y_true), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='Ideal Line')

# 添加网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predicted vs True Values (Random Forest-sDGP)')
plt.legend()
plt.show()
plt.savefig(fig_save_path,dpi=300,format='png')
