from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# 读取数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')

# 指定特征与目标
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

# 初始化弹性网络回归模型
elastic_net = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)

# 拟合模型
elastic_net.fit(x_train, y_train)

# 保存模型（可选）

joblib.dump(elastic_net, '../mlModel_evaluate/elastic_net-0924.pkl')