import joblib
import pandas as pd
import data_process as dp
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 读取训练数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建岭回归模型
ridge_reg = Ridge(alpha=1.2)

# 训练模型
ridge_reg.fit(X_train, y_train)

# 保存模型
joblib.dump(ridge_reg, '../mlModel_evaluate/rigid-0924.pkl')