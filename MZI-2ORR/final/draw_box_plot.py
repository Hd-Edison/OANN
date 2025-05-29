import os
import pandas as pd
import matplotlib.pyplot as plt


def db_to_linear(db_value):
    if isinstance(db_value, list):
        return [10 ** (value / 10) for value in db_value]
    else:
        return 10 ** (db_value / 10)


# 假设文件夹路径
folders = [
    # "Clipped_ReLU_alpha=0.8_beta=0.75",
    "Sine_alpha=1.0_beta=-1.5707963267948966",
    "Softplus_alpha=15.0_beta=0.2",
    "GeLU_alpha=6.0_beta=0.25",
    "Quadratic",
    "SiLU_alpha=20.0_beta=0.25",
    "Parametric_ReLU_alpha=0.2_beta=0.15",
    "Exponential_beta=3.0",
    "ReLU0.15",
    "Sigmoid_alpha=10.0_beta=0.7"
]

# 更新的横坐标文本
x_labels = [
    # "Clipped_ReLU_0.8_0.75",
    "Sine_1.0_-π/2",
    "Softplus_15.0_0.2",
    "GeLU_6.0_0.25",
    "Quadratic",
    "SiLU_20.0_0.25",
    "Parametric_ReLU_0.2_0.15",
    "Exponential_3.0",
    "ReLU_0.15",
    "Sigmoid_10.0_0.7"
]

# 遍历文件夹，提取rmse_robust_0.010000.csv文件中的第一列数据，并将其除以output.txt中的rmse值
for file_name in ("kr1", "kr2", "k1", "k2", "phi", "a", "b", "phir2"):
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
        relative_changes = abs((df.iloc[:, 0].dropna().values - rmse_value) / rmse_value) \
                           * 100
        data.append(relative_changes)

    # 设置字体为 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # 创建箱型图
    fig, ax = plt.subplots()

    ax.boxplot(data, patch_artist=True)

    # 设置x轴标签
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontname='Times New Roman', fontsize=14)

    # 设置y轴标签
    ax.set_ylabel('RMSE (%)', fontname='Times New Roman', fontsize=14)

    # 添加网格
    ax.grid(True)

    # 调整布局以防止文本重叠
    plt.tight_layout()

    # 检查文件夹是否存在
    if not os.path.exists("robustness"):
        # 创建文件夹
        os.makedirs("robustness")
    # 保存图表
    plt.savefig('robustness/boxplot_%s.png' % (file_name))

    plt.close()
    del data
