import xgboost as xgb
from sklearn.model_selection import train_test_split

import data_process as dp
import joblib

# 加载数据
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义 XGBoost 回归模型
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.001,
                            max_depth = 5, alpha = 10, n_estimators = 10)

# 训练模型
model.fit(X_train, y_train)

# 输出模型
joblib.dump(model, '/.venv/sDGP_model/xgboost-0513.pkl')