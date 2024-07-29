from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import data_process as dp
from tqdm import tqdm

# 建立数据集
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
#y = data1[['sDGP']]
y = data1[['sUDI']]

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
    print(f"Iteration {i+1}, Training Score: {train_score}")

# 输出训练过程中每棵树的得分
print("Training Scores for each tree:", train_scores)

# 拟合模型
random_forest.fit(x_train, y_train)

# 保存模型
import joblib
joblib.dump(random_forest, 'D:\pythonProject\pythonProject\.venv\models\sUDI_model/random_forest_model-V1-0515.pkl')