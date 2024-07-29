import data_process as dp
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models,regularizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# 建立数据集
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
#y = data1[['sDGP']]
y = data1[['sUDI']]

#创建RBF神经网络模型
def rbf_model(input_shape, l2_reg):
    model = models.Sequential([
        layers.Dense(5, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dense(1, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))
    ])
    return model

# 定义 K 折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=45)

# 用于存储每次交叉验证的结果
cv_scores = []

# 执行 K 折交叉验证
for train_index, val_index in kfold.split(x,y):# train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标

    # 划分训练集和验证集
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 创建模型
    model = rbf_model(input_shape=x_train.shape[1], l2_reg=0.01)

    # 编译模型
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # 训练模型
    model.fit(x_train, y_train, epochs=20, batch_size=512,
              validation_data=(x_val, y_val))

    # 在验证集上评估模型性能
    scores = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation MSE: {scores[0]}, Validation MAE: {scores[1]}")

    # 将每次交叉验证的结果添加到列表中
    cv_scores.append(scores)

# 打印平均性能指标
cv_scores = np.array(cv_scores)
print(f"Average Validation MSE: {np.mean(cv_scores[:, 0])}, Average Validation MAE: {np.mean(cv_scores[:, 1])}")

# 保存整个模型
model.save('mlp_sUDI(kFold)-V1-0515.keras')