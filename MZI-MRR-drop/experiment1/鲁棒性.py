import os
import re
import numpy as np
import tensorflow as tf
from main import OANN
from activation_functions import *
import csv


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


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def robust(true_function, name, robust_index, parameter_to_robust, z_train=None):
    if not z_train:
        z_train = np.linspace(0, 1, 200)
    parameters = process_parameters(name)
    # 输出结果
    f = true_function.function(z_train)

    # 设置高斯噪声的参数
    mu = 0  # 噪声均值
    sigma = robust_index  # 噪声标准差
    rmse_dB = []
    data = np.array([parameters["k1"], parameters["k2"], parameters["kr1"], parameters["kr2"], parameters["phi"],
                     parameters["a"], parameters["b"], parameters["phir2"]])
    for i in range(100):
        # 生成高斯噪声

        noise = np.random.normal(mu, sigma, data.shape)
        # 将高斯噪声添加到原始数据
        k1_pred, k2_pred, kr1_pred, kr2_pred, phi_pred, a_pred, b_pred, phir2_pred = data

        if parameter_to_robust == "k1":
            k1_pred += noise[0]
        if parameter_to_robust == "k2":
            k2_pred += noise[1]
        if parameter_to_robust == "kr1":
            kr1_pred += noise[2]
        if parameter_to_robust == "kr2":
            kr2_pred += noise[3]
        if parameter_to_robust == "phi":
            phi_pred += noise[4]
        if parameter_to_robust == "a":
            a_pred += noise[5]
        if parameter_to_robust == "b":
            b_pred += noise[6]
        if parameter_to_robust == "phir2":
            phir2_pred += noise[7]
        k1_pred, k2_pred, kr1_pred, kr2_pred, phi_pred, a_pred, b_pred, phir2_pred = \
            clip_values(k1_pred), clip_values(k2_pred), clip_values(kr1_pred), clip_values(kr2_pred), clip_values(
                phi_pred, 0, 2 * np.pi), clip_values(a_pred, 0, 2 * np.pi), clip_values(b_pred, 0, 10), clip_values(
                phir2_pred, 0, 2 * np.pi),
        rmse_value = rmse(y_true=f,
                          y_pred=H(k1_pred, k2_pred, kr1_pred, kr2_pred, phi_pred, a_pred, b_pred, phir2_pred,
                                   z_train).numpy())
        rmse_dB.append(10 * np.log10(rmse_value))

    with open('./%s/rmse_robust_%s_%f.csv' % (name, parameter_to_robust, robust_index), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in rmse_dB:
            writer.writerow([item])

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
    for parameter_to_robust in ("k1", "k2", "kr1", "kr2", "phi", "a", "b", "phir2"):
        for name in names:
            if name == names[0]:
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ReLU(beta=beta)
                    robust(f, name=name + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[1]:
                for alpha in (0.6, 0.8, 1.0):
                    for beta in (0.55, 0.65, 0.75, 0.85):
                        f = Cliped_ReLU(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[2]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = ELu(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[3]:
                for alpha in (6., 8., 10.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = GeLu(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[4]:
                for alpha in (0.1, 0.15, 0.2):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = Parametric_ReLu(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[5]:
                for alpha in (10., 15., 20.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = SiLU(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[6]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.5, 0.6, 0.7):
                        f = Gaussian(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[7]:
                f = Quadratic(alpha=alpha, beta=beta)
                robust(f, name=name, robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[8]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Sigmoid(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[9]:
                for alpha in (1., np.pi, 2 * np.pi):
                    beta = (- np.pi / 2)
                    f = Sine(alpha=alpha, beta=beta)
                    robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[10]:
                for alpha in (10., 15., 20.):
                    for beta in (0.2, 0.3, 0.4):
                        f = Softplus(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[11]:
                for alpha in (5., 10., 15.):
                    for beta in (0.5, 0.7):
                        f = Tanh(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[12]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Softsign(alpha=alpha, beta=beta)
                        robust(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
            if name == names[13]:
                for beta in (3., 6., 9.):
                    f = Exponential(beta=beta)
                    robust(f, name=name + "_beta=" + str(beta), robust_index=robust_index, parameter_to_robust=parameter_to_robust)
