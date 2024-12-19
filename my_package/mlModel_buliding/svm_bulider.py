import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from tqdm import tqdm

# 读取数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')

# 指定特征与目标
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]

# 划分训练集和测试集
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.2, random_state=20)

# 创建支持向量机回归模型
model = SVR(kernel='rbf')

# 多次训练
n_iterations = 10  # 训练轮数
for i in tqdm(range(n_iterations), desc="Training SVR Model"):
    model.fit(x_train, y_train.values.ravel())

    # 在每次训练后进行预测并计算评估指标
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 输出评估结果
    print(f"Iteration {i + 1}: Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

# 保存模型
joblib.dump(model, 'svr_model-0924.pkl')

