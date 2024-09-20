import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

"""
功能：训练随机森林模型
使用步骤：
1.更改数据集输入路径
2.更改训练的特征值与预测值
3.更改输出路径
4.更改超参数（如必要，一般不用改）
5.运行程序等模型输出（训练时间约5分钟） -> 去“model_evaluate”文件夹验证模型
"""

model_output_path = '../source/model_optimizer/model_0920/sUDI_RF_0920.pkl'
# 建立数据集
df_normalized_path = r'../source/data/1126335/outside_0920/240920_normalized.csv'
df_normalized = pd.read_csv(df_normalized_path)

print(df_normalized.shape)

train_data = df_normalized

x = train_data[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = train_data[['sUDI']]

print(x.shape, y.shape)

# 将 y 转换为一维数组
y_array = y.values.ravel()

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y_array, test_size=0.2, random_state=42)

# 初始化随机森林回归模型
random_forest = RandomForestRegressor(n_estimators=15,
                                      min_samples_leaf=2,
                                      min_samples_split=5,
                                      max_depth=None,
                                      random_state=42)

# 进度条
# 迭代次数
n_estimators = random_forest.n_estimators
# 训练过程中的每棵树的得分列表
train_scores = []
# 使用 tqdm 创建进度条，并在循环中更新进度条
for i in tqdm(range(n_estimators), desc="Training Progress"):
    random_forest.fit(x_train, y_train)
    train_score = random_forest.score(x_train, y_train)
    train_scores.append(train_score)
    print(f"Iteration {i + 1}, Training Score: {train_score}")

# 输出训练过程中每棵树的得分
print("Training Scores for each tree:", train_scores)

# 拟合模型
random_forest.fit(x_train, y_train)

# 获取路径中不包括文件名的部分
directory = os.path.dirname(model_output_path)

# 如果路径不存在，则创建路径
if not os.path.exists(directory):
    os.makedirs(directory)
# 保存模型
joblib.dump(random_forest, model_output_path)
