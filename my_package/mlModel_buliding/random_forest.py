import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

# 配置日志
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def setup_logging():
    """打印日志到控制台"""
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


setup_logging()


def train_random_forest_model(aim_target, model_output_path, df_normalized_path):
    """训练随机森林模型并保存"""
    try:
        logging.info("Reading normalized dataset...")
        df_normalized = pd.read_csv(df_normalized_path)
        logging.info(f"Dataset shape: {df_normalized.shape}")

        x = df_normalized[['Azimuth', 'Altitude', 'Shade Angle', 'Shade Interval', 'Direct Radiation']]
        y = df_normalized[[f'{aim_target}']]

        logging.info(f"Features shape: {x.shape}, Target shape: {y.shape}")

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
        n_estimators = random_forest.n_estimators
        train_scores = []

        logging.info("Starting training with progress bar...")
        for i in tqdm(range(n_estimators), desc="Training Progress"):
            random_forest.fit(x_train, y_train)
            train_score = random_forest.score(x_train, y_train)
            train_scores.append(train_score)
            logging.info(f"Iteration {i + 1}, Training Score: {train_score}")

        logging.info("Training completed.")
        logging.info(f"Training Scores for each tree: {train_scores}")

        # 拟合模型
        random_forest.fit(x_train, y_train)

        # 创建输出目录
        directory = os.path.dirname(model_output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存模型
        joblib.dump(random_forest, model_output_path)
        logging.info(f"Model saved at {model_output_path}")

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise


if __name__ == "__main__":
    # 配置参数
    aim_location = 'sz'
    my_aim_target = 'sUDI'
    train_date = '250107'

    my_model_output_path = f'../source/model_optimizer/model_{aim_location}-{train_date}/{my_aim_target}_RF.pkl'
    my_df_normalized_path = f'../source/data/data_mlTrain/{aim_location}/{aim_location}_normalizedDataset.csv'

    train_random_forest_model(my_aim_target, my_model_output_path, my_df_normalized_path)
