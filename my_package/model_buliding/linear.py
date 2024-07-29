from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import data_process as dp
import joblib

data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 创建线性回归模型
model = LinearRegression()

# 使用训练数据拟合模型
model.fit(x_train, y_train)

# 保存模型到文件
joblib.dump(model, 'linear_regression_model.pkl')
 
 