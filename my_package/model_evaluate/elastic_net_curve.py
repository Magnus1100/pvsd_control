import numpy as np
import matplotlib.pyplot as plt
import joblib

import data_process as dp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#指定特征与目标
x = dp.normalized_data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = dp.new_df[['sDGP']]

# 划分测试集
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.0001, random_state=20)

# 加载模型
path_elastic = '/.venv/sDGP_model/elastic_net_model-V1-0513.pkl'
model = joblib.load(path_elastic)

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

# 绘制预测值和真实值的曲线
plt.plot(x, y_pred, label='Predicted', color='blue')
plt.plot(x, y_true, label='True', color='red')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Predicted vs True Values(elastic_net)')
plt.legend()
plt.show()