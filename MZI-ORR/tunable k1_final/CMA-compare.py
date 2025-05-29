from main import OANN
import cma
import numpy as np
import os
from activation_functions import *
import re
import csv
import pandas as pd


def compare_result():
    # 读取第一个文件并获取前两列数据
    df = pd.read_csv(f'CMA-ES//rmse_comparison_0.csv', skiprows=1, header=None)
    folder_list = df.iloc[:, 0].tolist()
    rmse_list = df.iloc[:, 2].tolist()

    # 初始化数据列表，data 将是一个二维列表，按行存储
    data = [[folder, rmse] for folder, rmse in zip(folder_list, rmse_list)]

    # 追加其他文件中的数据 (第三列及之后)
    for i in range(6):
        try:
            df = pd.read_csv(f'CMA-ES//rmse_comparison_{i}.csv', skiprows=1, header=None)
            cma_es_rmse = df.iloc[:, 3].tolist()
            for j in range(len(data)):
                data[j].append(cma_es_rmse[j])  # 将对应的 CMA-ES RMSE 追加到同一行
        except:
            break

    # 将结果保存为 CSV 文件
    csv_file = os.path.join(".", f'CMA-ES//result.csv')

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Folder', 'RMSE (dB)', "CMA-ES RMSE 1", "CMA-ES RMSE 2", "CMA-ES RMSE 3", "CMA-ES RMSE 4", "CMA-ES RMSE 5", "CMA-ES RMSE 6"])
        writer.writerows(data)

    print(f"Results saved to {csv_file}")



if __name__ == '__main__':
    compare_result()
