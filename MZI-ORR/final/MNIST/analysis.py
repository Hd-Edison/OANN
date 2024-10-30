import os
import pandas as pd

# 指定文件夹路径
folder_path = '.'  # 替换为你的文件夹路径

# 初始化一个空列表以存储结果
results = []

# 遍历文件夹中的所有文件
for csv_file in os.listdir(folder_path):
    if csv_file.endswith('.csv'):  # 只处理 .csv 文件
        file_path = os.path.join(folder_path, csv_file)

        # 读取 CSV 文件
        data = pd.read_csv(file_path)

        # 获取最后一行的最后两列
        last_row = data.iloc[-1, -2:]  # 获取最后一行的最后两列

        # 添加文件名和数据到结果列表
        results.append((csv_file, last_row[0], last_row[1]))

# 创建 DataFrame 来存储结果
results_df = pd.DataFrame(results, columns=['File Name', 'Value 1', 'Value 2'])

# 按第一个数据排序
sorted_results = results_df.sort_values(by='Value 1')

# 显示结果
print(sorted_results)
