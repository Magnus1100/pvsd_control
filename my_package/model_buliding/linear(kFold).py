from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import data_process as dp
import joblib

# 读取数据
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 创建k折对象（5折，洗牌）
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 用于存储每个折叠的 MSE
mse_scores = []

# 执行 K 折交叉验证
for train_index, val_index in kf.split(x,y):# train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标

    # 划分训练集和验证集
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 训练模型
    model.fit(x_train, y_train)

# 保存模型到文件
joblib.dump(model, 'D:\pythonProject\pythonProject\.venv\model(kFold)/linear(kFold)-V2.pkl')


