import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 加载数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data['sDGP']  # 转换为一维 Series

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)

# 定义 XGBoost 回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.001,
                         max_depth=5, alpha=10, n_estimators=10)

# 训练模型
model.fit(X_train, y_train, verbose=True)  # 输出训练过程

# 输出模型
joblib.dump(model, '../mlModel_evaluate/xgboost-0924.pkl')
