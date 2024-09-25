import numpy as np
import matplotlib.pyplot as plt
import joblib

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')
#指定特征与目标
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.0001, random_state=20)
path_xgboost = 'xgboost-0924.pkl'

# 加载模型
model = joblib.load(path_xgboost)

# 使用模型预测测试集
y_pred = model.predict(x_test)

# 创建序号
x = range(len(y_pred))

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
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# 绘制实际值和预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, label='Predicted vs Actual', color='blue')

# 绘制理想线 y = x
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal Line (y=x)')

# 添加标签和标题
plt.xlabel('Actual sDGP')
plt.ylabel('Predicted sDGP')
plt.title('Predicted vs Actual sDGP')
plt.legend()
plt.grid()

# 显示图形
plt.show()