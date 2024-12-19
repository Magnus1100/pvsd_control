import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')

# 指定特征与目标
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

# 划分训练集和测试集
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.2, random_state=20)

# 创建多项式特征
degree = 2  # 选择多项式的阶数
poly = PolynomialFeatures(degree)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(x_train_poly, y_train)

# 保存模型
joblib.dump(model, 'polynomial_regression_model.pkl')

# 预测
y_pred = model.predict(x_test_poly)

# 计算均方误差（MSE）和 R-squared
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
# 计算均方根误差（RMSE）
rmse = np.sqrt(mse)
# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2 Score): {r2:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

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