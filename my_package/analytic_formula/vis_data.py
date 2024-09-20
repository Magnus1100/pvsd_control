import numpy as np
import pandas as pd
import os

vis_path = r'../source/data/vis_dataset_outside.txt'

def main():

    vis = np.loadtxt(vis_path)
    sd_position = np.linspace(-0.14, 0.14, 29)
    sd_angle = np.linspace(0, 90, 91)

    copied_angle = []
    copied_position = []

    # 循环复制
    for i in range(29):
        for item in sd_angle:
            copied_angle.append(item)

    # 单次复制
    copied_position = np.repeat(sd_position, 91)
    vis_dataset = pd.DataFrame({
        'vis':vis
    })
    vis_dataset['sd_angle'] = copied_angle
    vis_dataset['sd_position'] = copied_position.round(2)
    vis_dataset.to_csv('vis_data_outside_0920.csv')
    print(vis_dataset)


if __name__ == '__main__':
    main()
