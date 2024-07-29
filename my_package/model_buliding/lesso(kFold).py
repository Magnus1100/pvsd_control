import data_process as dp
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
import joblib

# 建立数据集
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 创建k折对象（5折，洗牌）
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化 Lasso 模型
lasso_model = Lasso(alpha=0.1)  # 这里可以调整 Lasso 模型的超参数 alpha

# 执行 K 折交叉验证
for train_index, val_index in kf.split(x,y):# train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标

    # 划分训练集和验证集
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 训练模型
    lasso_model.fit(x_train, y_train)

# 保存模型到文件
joblib.dump(lasso_model, 'D:\pythonProject\pythonProject\.venv\model(kFold)/lasso(kFold)-V1-0513.pkl')
