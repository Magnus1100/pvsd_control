from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
import data_process as dp

# 建立数据集
data1 = dp.new_df
data2 = dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 将 y 转换为一维数组
y_array = y.values.ravel()

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y_array, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [5,10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_depth': [10, 15, 20, 25, None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# 初始化随机森林回归模型
random_forest = RandomForestRegressor(random_state=42)

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# 进度条
print("Starting Grid Search...")
grid_search.fit(x_train, y_train)

# 获取最佳参数和最佳得分
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# 输出最佳参数和得分
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# 输出测试集上的得分
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)

print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")

# 保存最佳模型
joblib.dump(best_model, '/.venv/sDGP_model/sDGP_random_forest_model.pkl')