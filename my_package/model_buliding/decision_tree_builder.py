import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 加载数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data['sDGP']  # 将目标变量转换为一维 Series

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)

# 定义决策树回归模型
model = DecisionTreeRegressor(random_state=60)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# 保存模型
joblib.dump(model, '../model_evaluate/decision_tree_model_0924.pkl')