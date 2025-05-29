import os
import re
import numpy as np
import tensorflow as tf
from main import OANN
from activation_functions import *
import csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


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
    return abs(transfer_function_MZI_ORR(HR=oann.transfer_function_MRR(k=kr, gamma=0.74, neff=2.389,
                                                                       phi=oann.calculate_phi(a=a, b=b,
                                                                                              z_squared=z_squared),
                                                                       L=103.26e-6 * 2 * np.pi),
                                         k1=k1, k2=k2, phi=phi)) ** 2 * z_squared
def plot_loss(loss, name=None):
    plt.plot(loss, linewidth=3, linestyle='--', label="Training loss")
    plt.xlabel('Epochs', fontname='Times New Roman', fontsize=18)
    plt.ylabel('Loss', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=18)
    plt.yticks(fontfamily='Times New Roman', fontsize=18)
    font = FontProperties(family='Times New Roman', size=18)
    plt.legend( prop=font)
    plt.tight_layout()
    plt.savefig("./%s/loss.png" % name)
    plt.close()


def plot_prediction_curve(z_squared, function, predicted_function, name):
    plt.plot(z_squared, function, linewidth=3, label="True function")
    plt.plot(z_squared, predicted_function, linewidth=3, linestyle='--', label="Prediction")
    plt.xlabel('Normalized input power', fontname='Times New Roman', fontsize=18)
    plt.ylabel('Normalized output power', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=18)
    plt.yticks(fontfamily='Times New Roman', fontsize=18)
    font = FontProperties(family='Times New Roman', size=18)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig('./%s/prediction.png' % name)
    plt.close()

def plot(true_function, name, robust_index, z_train=None):
    if not z_train:
        z_train = np.linspace(0, 1, 200)
    parameters = process_parameters(name)
    # 输出结果
    print("Predicted parameters:")
    f = true_function.function(z_train)
    # 读取CSV文件
    df = pd.read_csv('./%s/loss_OANN_model.csv' % name)
    # 提取loss列的数据（从第二行开始）
    loss = df['Training Loss'].values[1:]  # 从第二行开始读取

    plot_prediction_curve(z_train, function=f,
                          predicted_function=H(parameters["kr"], parameters["phi"], parameters["a"], parameters["b"],
                                               parameters["k1"], parameters["k2"], z_train), name=name)
    plot_loss(loss, name)
def clip_values(values, min_value=0, max_value=1):
    """
    将数值裁剪到 [min_value, max_value] 范围内。

    参数:
    values (array-like): 要裁剪的数值。
    min_value (float): 最小值。
    max_value (float): 最大值。

    返回:
    array-like: 裁剪后的数值。
    """
    return np.clip(values, min_value, max_value)

if __name__ == '__main__':
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            parameters = process_parameters(dir_name)
            subfolder_path = os.path.join(root, dir_name)
            output_file_path = os.path.join(subfolder_path, 'output.txt')
            csv_output_path = os.path.join(subfolder_path, 'rmse_data.csv')


    names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
             "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
    robust_index = 0.01
    for name in names:
        if name == names[0]:
            for beta in (0.15, 0.25, 0.35, 0.55):
                f = ReLU(beta=beta)
                plot(f, name=name + str(beta), robust_index=robust_index)
        if name == names[1]:
            for alpha in (0.6, 0.8, 1.0):
                for beta in (0.55, 0.65, 0.75, 0.85):
                    f = Cliped_ReLU(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[2]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ELu(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[3]:
            for alpha in (6., 8., 10.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = GeLu(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[4]:
            for alpha in (0.1, 0.15, 0.2):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = Parametric_ReLu(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[5]:
            for alpha in (10., 15., 20.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = SiLU(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[6]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.5, 0.6, 0.7):
                    f = Gaussian(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[7]:
            f = Quadratic(alpha=alpha, beta=beta)
            plot(f, name=name, robust_index=robust_index)
        if name == names[8]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Sigmoid(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[9]:
            for alpha in (1., np.pi, 2 * np.pi):
                beta = (- np.pi / 2)
                f = Sine(alpha=alpha, beta=beta)
                plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[10]:
            for alpha in (10., 15., 20.):
                for beta in (0.2, 0.3, 0.4):
                    f = Softplus(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[11]:
            for alpha in (5., 10., 15.):
                for beta in (0.5, 0.7):
                    f = Tanh(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[12]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Softsign(alpha=alpha, beta=beta)
                    plot(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index)
        if name == names[13]:
            for beta in (3., 6., 9.):
                f = Exponential(beta=beta)
                plot(f, name=name + "_beta=" + str(beta), robust_index=robust_index)
