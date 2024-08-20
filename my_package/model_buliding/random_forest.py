import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 建立数据集
df_path = r'../source/data/1126335/240820_sDGP.csv'
df = pd.read_csv(df_path)

df_normalized_path = r'../source/data/1126335/240820_normalized_sDGP.csv'
df_normalized = pd.read_csv(df_normalized_path)

data1 = df_normalized
data2 = df

x = data1[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data2[['sDGP']]

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
model_path = r'D:\03-GitHub\pvsd_control\my_package\source\models\sDGP_RF_0820_V1.pkl'

# 获取路径中不包括文件名的部分
directory = os.path.dirname(model_path)

# 如果路径不存在，则创建路径
if not os.path.exists(directory):
    os.makedirs(directory)
# 保存模型
joblib.dump(random_forest, model_path)
