import joblib
import numpy as np
import data_process as dp
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 读取训练数据
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建岭回归模型
ridge_reg = Ridge(alpha=1.2)

# 训练模型
ridge_reg.fit(X_train, y_train)

# 保存模型
joblib.dump(ridge_reg, 'D:\pythonProject\pythonProject\.venv\model/ridge-1.2-0514.pkl')