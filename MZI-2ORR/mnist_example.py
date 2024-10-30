import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from main import OANN
import numpy as np
import os
import re
import pandas as pd

def extract_parameters(file_path):
    # 定义正则表达式以匹配所有参数
    pattern = re.compile(
        r'kr: ([\d.e-]+),\s*phi: ([\d.e-]+),\s*a: ([\d.e-]+),\s*b: ([\d.e-]+),\s*k1: ([\d.e-]+),\s*k2: ([\d.e-]+)',
        re.DOTALL)
    rmse_pattern = re.compile(r'rmse = ([-\d.,\s]+) dB', re.DOTALL)

    parameters = {}
    rmse_values = []

    with open(file_path, 'r') as file:
        content = file.read()

        # 提取参数
        param_match = pattern.search(content)
        if param_match:
            parameters['kr'] = float(param_match.group(1))
            parameters['phi'] = float(param_match.group(2))
            parameters['a'] = float(param_match.group(3))
            parameters['b'] = float(param_match.group(4))
            parameters['k1'] = float(param_match.group(5))
            parameters['k2'] = float(param_match.group(6))

    return parameters, np.array(rmse_values)


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters

def H(kr, phi, a, b, k1, k2, z_squared):
    oann = OANN(lam=1.55)
    return abs(oann.transfer_function_MZI_ORR(HR=oann.transfer_function_ORR(
        k=kr, gamma=0.74, neff=2.389, phi=oann.calculate_phi(a=a * 2 * np.pi, b=b * 10, z_squared=z_squared), L=103.26e-6 * 2 * np.pi),
        k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared


# 定义自定义的非线性激活函数
def custom_activation(x):

    return H(kr=parameters["kr"], phi=parameters["phi"], a=parameters["a"], b=parameters["b"], k1=parameters["k1"], k2=parameters["k2"], z_squared=x)

if __name__ == '__main__':
    # 设置文件夹路径和输出 CSV 文件路径
    names = ["Clipped_ReLU_alpha=0.8_beta=0.65",
             "Sine_alpha=1.0_beta=-1.5707963267948966",
             "Softplus_alpha=10.0_beta=0.4",
             "Sigmoid_alpha=10.0_beta=0.7",
             "Quadratic",
             "SiLU_alpha=10.0_beta=0.35",
             "Parametric_ReLU_alpha=0.2_beta=0.35",
             "GeLU_alpha=10.0_beta=0.45",
             "Exponential_beta=3.0",
             "ReLU0.55",
             "Tanh_alpha=5.0_beta=0.7"]
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
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=256, verbose=2)

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
        df.to_csv(folder_path + 'training_history.csv', index=False)

        # 评估模型
        scores = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test loss: {scores[0]}")
        print(f"Test accuracy: {scores[1]}")

        # 绘制训练过程中的准确率和损失
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='test accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig('%s/training.png' % folder_path)
        plt.close()
