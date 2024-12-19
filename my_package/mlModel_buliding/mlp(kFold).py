import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# 读取数据
data = pd.read_csv('../source/data/1126335/outside_0920/240920_normalized.csv')

# 指定特征与目标
x = data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data[['sDGP']]


# 创建 RBF 神经网络模型
def rbf_model(input_shape, l2_reg):
    model = keras.Sequential([
        layers.Dense(5, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(l2_reg))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# 定义 K 折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=45)

mse_scores = []
r2_scores = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = rbf_model(input_shape=x.shape[1], l2_reg=0.01)

    # 设置 verbose=1 以输出训练进度
    model.fit(x_train, y_train, epochs=10, batch_size=3, verbose=1)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

# 输出平均分数
print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Mean R-squared: {np.mean(r2_scores):.4f}")

# 保存整个模型
model.save('mlp_sDGP(kFold)-0924.keras')