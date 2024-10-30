import matplotlib.pyplot as plt
import os
from pandas import pd
def plot_training_data(folder_path):
    # 设置字体和大小
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 图例标签
    x_labels = [
        "Clipped_ReLU_0.8_0.65",
        "Sine_1.0_-π/2",
        "Softplus_15.0_0.4",
        "Sigmoid_10.0_0.7",
        "Quadratic",
        "SiLU_10.0_0.35",
        "Parametric_ReLU_0.2_0.35",
        "GeLU_6.0_0.25",
        "Exponential_3.0",
        "ReLU_0.25",
        "Tanh_5.0_0.7"
    ]

    # 检查文件数量和图例标签数量是否一致
    if len(csv_files) != len(x_labels):
        raise ValueError("CSV文件数量和图例标签数量不一致")

    # 初始化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for csv_file, x_label in zip(csv_files, x_labels):
        # 构造文件路径
        file_path = os.path.join(folder_path, csv_file)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 绘制 loss 图
        axes[0].plot(data['epoch'], data['loss'], label=x_label)
        axes[0].set_ylabel('Loss', fontsize=14, fontweight='bold', fontname='Times New Roman')

        # 绘制 val_loss 图
        axes[1].plot(data['epoch'], data['val_loss'], label=x_label)
        axes[1].set_ylabel('Validation Loss', fontsize=14, fontweight='bold', fontname='Times New Roman')

        # 绘制 accuracy 图
        axes[2].plot(data['epoch'], data['accuracy'], label=x_label)
        axes[2].set_ylabel('Accuracy', fontsize=14, fontweight='bold', fontname='Times New Roman')

        # 绘制 val_accuracy 图
        axes[3].plot(data['epoch'], data['val_accuracy'], label=x_label)
        axes[3].set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold', fontname='Times New Roman')

    # 获取所有子图的handles和labels
    handles, labels = axes[0].get_legend_handles_labels()

    # 在图形下方添加图例
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=6, fontsize=12, title_fontsize=14,
               frameon=True)

    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    # 调整布局，使图例不会与子图重叠
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=1., bottom=0.15, wspace=0.1, hspace=0.2)

    # 保存图像
    plt.savefig(os.path.join(folder_path, "figure.png"))
    plt.show()


if __name__ == '__main__':
    # 设置包含CSV文件的文件夹路径
    folder_path = 'MNIST\\'  # 替换为实际的文件夹路径

    # 调用函数绘制图表
    plot_training_data(folder_path)