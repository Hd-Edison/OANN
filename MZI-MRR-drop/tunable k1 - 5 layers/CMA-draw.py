import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置两个文件夹路径
folder_path_lra_cma = 'LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0'  # LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0 文件夹路径
folder_path_cma_es = 'CMA-ES-sigma=0.5-2.0'  # CMA-ES 文件夹路径
number_to_draw = 15
# 2. 正则表达式匹配sigma=xxx
sigma_pattern = re.compile(r'sigma=([\d.]+)')  # 匹配 sigma=xxx，xxx 是一个浮点数

# 3. 获取文件夹下所有以sigma=xxx结尾的csv文件
csv_files_lra_cma = [f for f in os.listdir(folder_path_lra_cma) if f.endswith('.csv') and sigma_pattern.search(f)]
csv_files_cma_es = [f for f in os.listdir(folder_path_cma_es) if f.endswith('.csv') and sigma_pattern.search(f)]

# 4. 创建一个绘图
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))

# 5. 定义颜色映射（colormap），根据 sigma 值分配不同颜色
num_colors = max(len(csv_files_lra_cma), len(csv_files_cma_es))
cmap = plt.get_cmap('viridis', num_colors)  # 使用 'viridis' colormap，可以更换成其他 colormap


def clean_sigma(sigma_str):
    """清理 sigma 字符串，将多余的点去掉并转换为浮点数"""
    sigma_str = sigma_str.rstrip('.')  # 移除末尾的多余点
    try:
        return float(sigma_str)
    except ValueError:
        return None  # 如果无法转换为 float，则返回 None

# original data
data = pd.read_csv("rmse_values.csv", skiprows=1, header=None)  # header=None 用于没有列名的情况

raw_data = data.iloc[:, 1]
raw_data = raw_data[raw_data != '[]']
sorted_data = raw_data.astype(float).sort_values()

color = cmap(0.5)
cma_style = {'color': color, 'linestyle': 'dotted'}  # LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0 文件夹的样式

# 绘制排序后的曲线
plt.plot(sorted_data.values[:number_to_draw], label=f'ANN result', **cma_style)

# 6. 遍历LRA-CMA文件夹中的csv文件并读取第四列
for i, csv_file in enumerate(csv_files_lra_cma):
    file_path = os.path.join(folder_path_lra_cma, csv_file)

    # 使用正则表达式提取 sigma 值
    sigma_match = sigma_pattern.search(csv_file)

    if sigma_match:
        sigma_value = clean_sigma(sigma_match.group(1))  # 提取 sigma 值，转换为 float

        # 读取csv文件，跳过第一行
        data = pd.read_csv(file_path, skiprows=1, header=None)  # header=None 用于没有列名的情况

        # 检查文件是否有至少四列
        if data.shape[1] >= 4:
            # 提取第四列，并按升序排序
            # 先筛选出非 '[]' 的值
            raw_data = data.iloc[:, 3]
            raw_data = raw_data[raw_data != '[]']
            sorted_data = raw_data.astype(float).sort_values()  # 使用列索引 3 来获取第四列

            color = cmap(i / num_colors)
            lra_cma_style = {'color': color, 'linestyle': '-'}  # LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0 文件夹的样式

            # 绘制排序后的曲线
            plt.plot(sorted_data.values[:number_to_draw], label=f'LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0 σ={sigma_value}', **lra_cma_style)
        else:
            print(f"文件 {csv_file} 中没有足够的列")

# 7. 遍历CMA-ES-new文件夹中的csv文件并读取第四列
for i, csv_file in enumerate(csv_files_cma_es):
    file_path = os.path.join(folder_path_cma_es, csv_file)

    # 使用正则表达式提取 sigma 值
    sigma_match = sigma_pattern.search(csv_file)
    if sigma_match:
        sigma_value = clean_sigma(sigma_match.group(1))  # 提取 sigma 值，转换为 float

        # 读取csv文件，跳过第一行
        data = pd.read_csv(file_path, skiprows=1, header=None)  # header=None 用于没有列名的情况

        # 检查文件是否有至少四列
        if data.shape[1] >= 4:
            raw_data = data.iloc[:, 3]
            raw_data = raw_data[raw_data != '[]']
            sorted_data = raw_data.astype(float).sort_values()  # 使用列索引 3 来获取第四列

            color = cmap(i / num_colors)
            cma_es_style = {'color': color, 'linestyle': '--'}  # CMA-ES 文件夹的样式
            # 绘制排序后的曲线
            plt.plot(sorted_data.values[:15], label=f'CMA-ES σ={sigma_value}', **cma_es_style)
        else:
            print(f"文件 {csv_file} 中没有足够的列")

# 8. 设置图例、标题和坐标轴标签
# plt.title(f'LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0 vs CMA-ES')

plt.xlabel('Index of Activation Functions')  # 可以修改为具体的横轴含义
plt.ylabel('RMSE (dB)')  # 将纵轴设为 RMSE (dB)
import numpy

y_min, y_max = plt.ylim()  # 获取当前纵轴范围
yticks = numpy.arange(y_min, y_max, 5)  # 每隔 5 个点取一个刻度
plt.yticks(yticks)
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.tight_layout()

# 9. 显示绘制的图
plt.savefig(f'./CMA-ES/Comparation_{number_to_draw}.png')
plt.close()
