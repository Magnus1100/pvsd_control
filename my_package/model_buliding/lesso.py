import tensorflow as tf
import data_process as dp
from sklearn.model_selection import train_test_split

# 建立数据集
data1 = dp.new_df
data2= dp.normalized_data

x = data2[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval']]
y = data1[['sDGP']]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义Lasso回归模型
inputs = tf.keras.Input(shape=(X_train.shape[1],))
outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.01))(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# 评估模型
loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)

# 输出模型
model.save('D:\pythonProject\pythonProject\.venv\model/lesso-V2-0513.keras')