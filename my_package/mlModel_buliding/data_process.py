import pandas as pd
import numpy as np
import math as mt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# 配置日志
logging.basicConfig(
    filename='data_processing.log',
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


def normalize_data(data):
    """标准化数据"""
    try:
        logging.info("Normalizing data...")
        # 提取列名
        column_names = data.columns

        # 归一化数据
        scaler = MinMaxScaler()
        normalized_dataset = scaler.fit_transform(data)

        # 将归一化后的数据转换为 DataFrame，并将列名添加回来
        normalized_df = pd.DataFrame(normalized_dataset, columns=column_names)
        logging.info("Data normalization completed successfully.")
        return normalized_df
    except Exception as e:
        logging.error(f"Error in normalizing data: {e}")
        raise


def process_and_save_data(epw_path, sDGP_path, sUDI_path, angle_path, position_path, output_path, normal_output_path):
    """处理并保存数据"""
    try:
        # 读取数据文件
        logging.info("Reading input files...")
        epw_data = pd.read_csv(epw_path)
        azimuth = epw_data['Azimuth']
        altitude = epw_data['Altitude']
        direct_rad = epw_data['Direct_Rad']

        sDGP = np.loadtxt(sDGP_path)
        sUDI = np.loadtxt(sUDI_path)

        sd_angle = np.loadtxt(angle_path)
        sd_angle = [round(mt.radians(angle), 2) for angle in sd_angle]
        sd_position = np.loadtxt(position_path)

        # 复制交替的列数据 255 次
        logging.info("Duplicating azimuth, altitude, and radiation data...")
        copied_azimuth = np.tile(azimuth, 255).tolist()
        copied_altitude = np.tile(altitude, 255).tolist()
        copied_direct_rad = np.tile(direct_rad, 255).tolist()

        # 重复复制遮阳数据 4417 次
        logging.info("Repeating shade angle and interval data...")
        copied_angle = np.repeat(sd_angle, 4417)
        copied_interval = np.repeat(sd_position, 4417)

        # 创建新的 DataFrame
        logging.info("Creating new DataFrame...")
        new_df = pd.DataFrame({
            'Azimuth': copied_azimuth,
            'Altitude': copied_altitude,
            'Shade Angle': copied_angle,
            'Shade Interval': copied_interval,
            'Direct Radiation': copied_direct_rad,
            'sDGP': sDGP,
            'sUDI': sUDI
        })

        # 保存数据
        new_df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")

        # 标准化数据并保存
        normalized_data = normalize_data(new_df)
        normalized_data.to_csv(normal_output_path, index=False)
        logging.info(f"Normalized data saved to {normal_output_path}")

    except Exception as e:
        logging.error(f"Error in processing data: {e}")
        raise


# 调用主函数
if __name__ == "__main__":
    aim_location = 'sz'
    my_epw_path = f'../source/data/data_shadeCalculate/{aim_location}/epwData_{aim_location}.csv'
    my_sDGP_path = f'../source/data/data_mlTrain/{aim_location}/{aim_location}_sDGP_nearWindow.txt'
    my_sUDI_path = f'../source/data/data_mlTrain/{aim_location}/{aim_location}_sUDI.txt'
    my_angle_path = '../source/data/data_mlTrain/angle_255.txt'
    my_position_path = '../source/data/data_mlTrain/position_255.txt'

    # 输出路径由主函数外部配置
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
    my_output_path = f'../source/data/data_mlTrain/{aim_location}/{aim_location}_mlDataset_{current_datetime}.csv'
    my_normal_output_path = f'../source/data/data_mlTrain/{aim_location}/{aim_location}_normalizedDataset_{current_datetime}.csv'

    process_and_save_data(
        my_epw_path,
        my_sDGP_path,
        my_sUDI_path,
        my_angle_path,
        my_position_path,
        my_output_path,
        my_normal_output_path
    )
