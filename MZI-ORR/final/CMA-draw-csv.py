import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# 1. 设置文件夹路径
folder_path_lra_cma = 'LRA-CMA-sigma=0.5-2.0'
folder_path_cma_es = 'CMA-ES-sigma=0.5-2.0'
number_to_draw = -1

# 2. 正则表达式匹配 sigma 值
sigma_pattern = re.compile(r'sigma=([\d.]+)')

# 3. 获取匹配文件
csv_files_lra_cma = [f for f in os.listdir(folder_path_lra_cma) if f.endswith('.csv') and sigma_pattern.search(f)]
csv_files_cma_es = [f for f in os.listdir(folder_path_cma_es) if f.endswith('.csv') and sigma_pattern.search(f)]

# 4. 设置绘图
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
point_size = 10
num_colors = max(len(csv_files_lra_cma), len(csv_files_cma_es))

# 5. 定义辅助函数
def format_label(label):
    replacements = {
        'alpha': 'α', 'beta': 'β',
        '1.5707963267948966': 'π/2', '3.141592653589793': 'π', '6.283185307179586': '2π',
        'Clipped_ReLU': 'CR', "Gaussian": "G", 'Parametric_ReLU': 'PR', "_": ", ",
    }
    for k, v in replacements.items():
        label = label.replace(k, v)
    return label

def clean_sigma(sigma_str):
    sigma_str = sigma_str.rstrip('.')
    try:
        return float(sigma_str)
    except ValueError:
        return None

# 6. 收集结果用字典
results_dict = defaultdict(dict)

# 7. 读取 ANN 数据
ann_data = pd.read_csv("rmse_values.csv", skiprows=1, header=None)
raw_data = ann_data.iloc[::2, 1]
raw_data = raw_data[raw_data != '[]']
ANN_data = raw_data.astype(float)
x_ann_labels = ann_data.iloc[::2, 0].astype(str)[raw_data.index]
x_ann_labels = [format_label(label) for label in x_ann_labels]

# 绘图 + 收集数据
cmap = plt.get_cmap('viridis', num_colors)
color = cmap(0.0)
ax.scatter(range(len(ANN_data.values[:number_to_draw])), ANN_data.values[:number_to_draw],
           label='ANN', color=color, s=point_size)
for x, y in zip(x_ann_labels[:number_to_draw], ANN_data.values[:number_to_draw]):
    results_dict[x]['ANN'] = y

# 8. 遍历 LRA-CMA-sigma=0.5-2.0 文件
for i, csv_file in enumerate(csv_files_lra_cma):
    file_path = os.path.join(folder_path_lra_cma, csv_file)
    sigma_match = sigma_pattern.search(csv_file)
    if sigma_match:
        sigma_value = clean_sigma(sigma_match.group(1))
        data = pd.read_csv(file_path, skiprows=1, header=None)
        if data.shape[1] >= 4:
            raw_data = data.iloc[::2, 3]
            raw_data = raw_data[raw_data != '[]']
            y_values = raw_data.astype(float).values[:number_to_draw]
            x_labels = data.iloc[::2, 0].astype(str)[raw_data.index]
            x_labels = [format_label(label) for label in x_labels][:number_to_draw]
            color = plt.get_cmap('tab20', num_colors)(i / num_colors)
            ax.scatter(x_labels, y_values, label=f'LRA-CMA-sigma=0.5-2.0, σ={sigma_value}', color=color, s=point_size)
            for x, y in zip(x_labels, y_values):
                results_dict[x][f'LRA-CMA-sigma=0.5-2.0, σ={sigma_value}'] = y
        else:
            print(f"文件 {csv_file} 中列数不足")

# 9. 遍历 CMA-ES 文件
for i, csv_file in enumerate(csv_files_cma_es):
    file_path = os.path.join(folder_path_cma_es, csv_file)
    sigma_match = sigma_pattern.search(csv_file)
    if sigma_match:
        sigma_value = clean_sigma(sigma_match.group(1))
        data = pd.read_csv(file_path, skiprows=1, header=None)
        if data.shape[1] >= 4:
            raw_data = data.iloc[::2, 3]
            raw_data = raw_data[raw_data != '[]']
            y_values = raw_data.astype(float).values[:number_to_draw]
            x_labels = data.iloc[::2, 0].astype(str)[raw_data.index]
            x_labels = [format_label(label) for label in x_labels][:number_to_draw]
            color = plt.get_cmap('coolwarm', num_colors)(i / num_colors)
            ax.scatter(x_labels, y_values, label=f'CMA-ES, σ={sigma_value}', color=color, s=point_size)
            for x, y in zip(x_labels, y_values):
                results_dict[x][f'CMA-ES, σ={sigma_value}'] = y
        else:
            print(f"文件 {csv_file} 中列数不足")

# 10. 设置图像
ax.set_ylabel('RMSE (dB)')
ax.set_yticks(np.arange(*ax.get_ylim(), step=5))
ax.legend(loc='lower right', ncol=2)
ax.grid(True)
ax.tick_params(axis="x", rotation=90)

# 11. 保存图像
fig.savefig(f'./CMA-ES/Comparation_{number_to_draw}.png')
plt.close()

# 12. 保存结果为 CSV
results_df = pd.DataFrame.from_dict(results_dict, orient='index')
results_df.index.name = 'x_label'
results_df = results_df.sort_index()
results_df.to_csv('CMA-ES/3D_scatter_results.csv')
