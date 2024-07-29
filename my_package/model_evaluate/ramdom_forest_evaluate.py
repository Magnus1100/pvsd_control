import numpy as np
import matplotlib.pyplot as plt
import data_process as dp
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 指定特征与目标
x = dp.normalized_data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = dp.new_df[['sDGP']]

# 划分测试集
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.0001, random_state=25)

# 加载模型
model_sdgp = joblib.load(r'../source/model_optimizer/sDGP_random_forest_model.pkl')

print(x_test)
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

# # 绘制预测值和真实值的曲线
# plt.figure(figsize=(15, 6))  # 设置高分辨率的长图
# plt.scatter(indices, y_pred, label='Predicted', color='blue', s=10)
# plt.scatter(indices, y_true, label='True', color='red', s=10)
#
# # 绘制误差线
# for i in indices:
#     plt.plot([i, i], [y_true[i], y_pred[i]], color='gray', linewidth=0.5)
#
# # 添加网格，增加网格密度和粗细
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # 设置 x 和 y 轴的刻度密度
# plt.xticks(np.arange(0, len(y_true), step=10))
#
# # 添加评价指标的文本到图外的右侧
# plt.figtext(0.98, 0.55, f'R^2: {r2:.3f}', fontsize=12, verticalalignment='top', horizontalalignment='right')
# plt.figtext(0.98, 0.50, f'MSE: {mse:.4f}', fontsize=12, verticalalignment='top', horizontalalignment='right')
# plt.figtext(0.98, 0.45, f'MAE: {mae:.3f}', fontsize=12, verticalalignment='top', horizontalalignment='right')

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

# 添加评价指标的文本到图外的右侧
plt.figtext(1, 0.55, f'R^2: {r2:.3f}', fontsize=12, verticalalignment='top', horizontalalignment='right')
plt.figtext(1, 0.50, f'MSE: {mse:.4f}', fontsize=12, verticalalignment='top', horizontalalignment='right')
plt.figtext(1, 0.45, f'MAE: {mae:.3f}', fontsize=12, verticalalignment='top', horizontalalignment='right')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predicted vs True Values (Random Forest-sDGP)')
plt.legend()
plt.show()
