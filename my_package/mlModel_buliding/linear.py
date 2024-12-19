from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 创建线性回归模型
model = LinearRegression()

# 使用训练数据拟合模型
model.fit(x_train, y_train)

# 保存模型到文件
joblib.dump(model, '../mlModel_evaluate/linear-0924.pkl')
 
 