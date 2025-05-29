import os
import pandas as pd
import matplotlib.pyplot as plt


def db_to_linear(db_value):
    if isinstance(db_value, list):
        return [10 ** (value / 10) for value in db_value]
    else:
        return 10 ** (db_value / 10)


folders = [
    # "Clipped_ReLU_alpha=0.8_beta=0.65",
    "Sine_alpha=1.0_beta=-1.5707963267948966",
    "Softplus_alpha=10.0_beta=0.4",
    "Sigmoid_alpha=10.0_beta=0.7",
    "Quadratic",
    "SiLU_alpha=10.0_beta=0.35",
    "Parametric_ReLU_alpha=0.2_beta=0.35",
    "GeLU_alpha=10.0_beta=0.45",
    "Exponential_beta=3.0",
    "ReLU0.55",
    "Tanh_alpha=5.0_beta=0.7"
]

# 更新的横坐标文本
x_labels = [
    # "Clipped_ReLU_0.8_0.65",
    "Sine_1.0_-pi/2",
    "Softplus_10.0_0.4",
    "Sigmoid_10.0_0.7",
    "Quadratic",
    "SiLU_10.0_0.35",
    "Parametric_ReLU_0.2_0.35",
    "GeLU_10.0_0.45",
    "Exponential_3.0",
    "ReLU0.55",
    "GeLU_5.0_0.7"
]

# 遍历文件夹，提取rmse_robust_0.010000.csv文件中的第一列数据，并将其除以output.txt中的rmse值
for file_name in ("kr", "phi", "a", "b", "k1", "k2"):
    file_name = "rmse_robust_" + file_name + "_0.010000.csv"
    # 初始化数据存储
    data = []
    for folder in folders:
        output_file = os.path.join(folder, "output.txt")
        rmse_robust_file = os.path.join(folder, file_name)

        # 提取output.txt中的rmse值
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "rmse =" in line:
                    rmse_value = float(line.split('=')[1].strip().split()[0])
                    break

        # 读取rmse_robust_0.010000.csv文件的第一列数据
        df = pd.read_csv(rmse_robust_file)
        rmse_linear = db_to_linear(rmse_value)
        relative_changes = abs((df.iloc[:, 0].dropna().values - rmse_value) / rmse_value) * 100
        data.append(relative_changes)

    # 设置字体为 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # 创建箱型图
    fig, ax = plt.subplots()

    ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel('RMSE (%)', fontname='Times New Roman', fontsize=16)
    ax.grid(True)
    plt.tight_layout()

    # 保存图表
    plt.savefig('robustness/boxplot_%s.png' % (file_name))

    plt.close()
    del data
