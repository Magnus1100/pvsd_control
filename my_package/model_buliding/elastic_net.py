from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import data_process as dp

# 建立数据集
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 初始化弹性网络回归模型
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# 拟合模型
elastic_net.fit(x_train, y_train)

# 保存模型（可选）
import joblib
joblib.dump(elastic_net, 'D:\pythonProject\pythonProject\.venv\model/elastic_net_model-V1-0513.pkl')