import os
import re
import csv


def find_rmse_value(file_path):
    # 定义匹配 rmse 的正则表达式
    rmse_pattern = re.compile(r'rmse = (-?\d+\.\d+) dB$')

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = rmse_pattern.search(line)
            if match:
                # 将捕获到的数字字符串转换为 float 类型
                return float(match.group(1))
    return None


def extract_rmse_to_csv(root_folder, output_csv):
    # 创建一个列表来保存结果
    results = []

    # 遍历根文件夹下的所有子文件夹
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            output_file_path = os.path.join(subfolder_path, 'output.txt')

            # 检查 output.txt 是否存在
            if os.path.isfile(output_file_path):
                rmse_value = find_rmse_value(output_file_path)
                if rmse_value is not None:
                    results.append([dir_name, rmse_value])
                    print(f"Found RMSE value {rmse_value} in folder {dir_name}")

    # 将结果保存到 CSV 文件
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder Name', 'RMSE Value'])
        csv_writer.writerows(results)

    print(f"Results written to {output_csv}")


# 使用该函数，设置根文件夹路径和输出 CSV 文件路径
root_folder = '.'
output_csv = 'rmse_values.csv'
extract_rmse_to_csv(root_folder, output_csv)

print(f"RMSE values have been extracted to {output_csv}")
