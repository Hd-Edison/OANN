from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from main import OANN
import numpy as np
import re
import pandas as pd


def extract_parameters(file_path):
    # 定义正则表达式以匹配所有参数
    pattern = re.compile(
        r'k1: ([\d.e-]+),\s*k2: ([\d.e-]+),\s*kr1: ([\d.e-]+),\s*kr2: ([\d.e-]+),\s*phi: ([\d.e-]+),\s*'
        r'a: ([\d.e-]+),\s*b: ([\d.e-]+),\s*phir2: ([\d.e-]+)',
        re.DOTALL)
    rmse_pattern = re.compile(r'rmse = ([-\d.,\s]+) dB', re.DOTALL)

    parameters = {}
    rmse_values = []

    with open(file_path, 'r') as file:
        content = file.read()

        # 提取参数
        param_match = pattern.search(content)
        if param_match:
            parameters['k1'] = float(param_match.group(1))
            parameters['k2'] = float(param_match.group(2))
            parameters['kr1'] = float(param_match.group(3))
            parameters['kr2'] = float(param_match.group(4))
            parameters['phi'] = float(param_match.group(5))
            parameters['a'] = float(param_match.group(6))
            parameters['b'] = float(param_match.group(7))
            parameters['phir2'] = float(param_match.group(8))

    return parameters, np.array(rmse_values)


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters


def H(k1, k2, kr1, kr2, phi, a, b, phir2, z_squared):
    oann = OANN(lam=1.55)
    return abs(oann.transfer_function_MZI_2MRR(HR1=oann.transfer_function_MRR(k=kr1, gamma=0.86, neff=2.389,
                                                                              phi=oann.calculate_phi(a * 2 * np.pi,
                                                                                                     b * 10,
                                                                                                     z_squared=z_squared),
                                                                              L=100e-6 * 2 * np.pi),
                                               HR2=oann.transfer_function_MRR(k=kr2, gamma=0.86, neff=2.389,
                                                                              phi=phir2 * 2 * np.pi,
                                                                              L=100e-6 * 2 * np.pi), k1=k1, k2=k2,
                                               phi=phi * 2 * np.pi)) ** 2 * z_squared


# 定义自定义的非线性激活函数
def custom_activation(x):
    return H(k1=parameters["k1"], k2=parameters["k2"], kr1=parameters["kr1"], kr2=parameters["kr2"],
             phi=parameters["phi"],
             a=parameters["a"], b=parameters["b"], phir2=parameters["phir2"], z_squared=x)


def plot_training_data(folder_path):
    # 设置字体和大小
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 更新的横坐标文本
    x_labels = [
        "Clipped_ReLU_1.0_0.55",
        "Sine_1.0_-π/2",
        "Softplus_15.0_0.4",
        "GeLU_6.0_0.35",
        "Quadratic",
        "SiLU_10.0_0.35",
        "Parametric_ReLU_0.15_0.35",
        "Exponential_3.0",
        "ReLU_0.25",
        "Tanh_10.0_0.7"
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
        # axes[0].set_ylabel('Loss')

        # 绘制 val_loss 图
        axes[1].plot(data['epoch'], data['val_loss'], label=x_label)
        # axes[1].set_ylabel('Validation Loss')

        # 绘制 accuracy 图
        axes[2].plot(data['epoch'], data['accuracy'], label=x_label)
        # axes[2].set_ylabel('Accuracy')

        # 绘制 val_accuracy 图
        axes[3].plot(data['epoch'], data['val_accuracy'], label=x_label)
        # axes[3].set_xlabel('Epoch')
        # axes[3].set_ylabel('Validation Accuracy')

    # 获取所有子图的handles和labels
    handles, labels = axes[0].get_legend_handles_labels()

    # 在图形下方添加图例
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=6)

    # 调整布局，使图例不会与子图重叠
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=1., bottom=0.15, wspace=0.1, hspace=0.2)

    # 保存图像
    plt.savefig(f"{folder_path}\\figure.png")
    plt.show()


if __name__ == '__main__':
    # 设置文件夹路径和输出 CSV 文件路径
    names = [
        "Clipped_ReLU_alpha=1.0_beta=0.55",
        "Sine_alpha=1.0_beta=-1.5707963267948966",
        "Softplus_alpha=15.0_beta=0.4",
        "GeLU_alpha=6.0_beta=0.35",
        "Quadratic",
        "SiLU_alpha=10.0_beta=0.35",
        "Parametric_ReLU_alpha=0.15_beta=0.35",
        "Exponential_beta=3.0",
        "ReLU0.25",
        "Sigmoid_alpha=10.0_beta=0.7"
    ]
    import os

    folder_name = "MNIST"

    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"'{folder_name}' already exists")
    for name in names:
        folder_path = "./" + name  # 替换为实际的文件夹路径

        # 执行函数，处理指定文件夹下的 output.txt 文件
        parameters = process_parameters(folder_path)

        # 加载 MNIST 数据集
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 数据预处理
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # 构建 CNN 模型
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation=custom_activation, input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation=custom_activation),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation=custom_activation),
            Dense(10, activation='softmax')
        ])

        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        # 训练模型
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=256, verbose=2)

        # 提取训练和验证过程中的loss和accuracy
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        accuracy = history_dict['accuracy']
        val_accuracy = history_dict['val_accuracy']

        # 创建DataFrame
        df = pd.DataFrame({
            'epoch': range(1, len(loss) + 1),
            'loss': loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'val_accuracy': val_accuracy
        })

        # 保存到CSV文件
        df.to_csv(folder_name + "\\" + folder_path + 'training_history.csv', index=False)

        # 评估模型
        scores = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test loss: {scores[0]}")
        print(f"Test accuracy: {scores[1]}")

        # 绘制训练过程中的准确率和损失
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['accuracy'], label='train accuracy')
        # plt.plot(history.history['val_accuracy'], label='test accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(history.history['loss'], label='train loss')
        # plt.plot(history.history['val_loss'], label='test loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        #
        # plt.savefig('%s/training.png' % folder_path)
        # plt.close()

    # 设置包含CSV文件的文件夹路径
    folder_path = 'MNIST\\'  # 替换为实际的文件夹路径

    # 调用函数绘制图表
    plot_training_data(folder_path)
