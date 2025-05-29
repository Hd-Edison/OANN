import os
import re
import numpy as np
from main import OANN
from activation_functions import *
import csv

oann = OANN(lam=1550e-9)

from numpy import pi
def extract_parameters(file_path):
    # 定义正则表达式以匹配所有参数
    pattern = re.compile(
        r'kr: ([\d.e-]+),\s*phi: ([\d.e-]+),\s*a: ([\d.e-]+),\s*b: ([\d.e-]+),\s*k1: ([\d.e-]+),\s*k2: ([\d.e-]+)',
        re.DOTALL)

    parameters = {}

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

    return parameters


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters = extract_parameters(output_file_path)
        return parameters


def H(kr, phi, a, b, k1, k2, z_squared=None):
    if z_squared is None:
        z_squared = tf.range(0, 1, 1 / 200)
    return abs(oann.transfer_function_MZI_MRR(HR=oann.transfer_function_MRR(k=kr, gamma=0.86, neff=2.389,
                                                                            phi=oann.calculate_phi(a=a, b=b,
                                                                                                   z_squared=z_squared),
                                                                            L=100e-6 * 2 * np.pi),
                                              k1=k1, k2=k2, phi=phi)) ** 2 * z_squared


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def robust(true_function, name, robust_index, parameter_to_robust, z_train=None):
    if not z_train:
        z_train = tf.range(0, 1, 1 / 200)
    parameters = process_parameters(name)
    if parameters is None:
        pass
    # 输出结果
    f = true_function.function()
    # 设置高斯噪声的参数
    mu = 0  # 噪声均值
    sigma = robust_index  # 噪声标准差
    rmse_dB = []
    data = np.array(
        [parameters["kr"], parameters["phi"], parameters["a"], parameters["b"], parameters["k1"], parameters["k2"]])
    for i in range(100):
        # 生成高斯噪声

        noise = np.random.normal(mu, sigma, data.shape)

        # 将高斯噪声添加到原始数据
        # kr_pred, phi_pred, a_pred, b_pred, k1_pred, k2_pred = data + noise

        # 试试只给一个参数添加噪声
        kr_pred, phi_pred, a_pred, b_pred, k1_pred, k2_pred = data
        # 初始化参数
        if parameter_to_robust == "kr":
            kr_pred += noise[0]
        if parameter_to_robust == "phi":
            phi_pred += noise[1]
        if parameter_to_robust == "a":
            a_pred += noise[2]
        if parameter_to_robust == "b":
            b_pred += noise[3]
        if parameter_to_robust == "k1":
            k1_pred += noise[4]
        if parameter_to_robust == "k2":
            k2_pred += noise[5]

        kr_pred, phi_pred, a_pred, b_pred, k1_pred, k2_pred = clip_values(kr_pred), clip_values(phi_pred, 0,
                                                                                                2 * np.pi), clip_values(
            a_pred, 0, 2 * np.pi), \
            clip_values(b_pred, 0, 10), clip_values(k1_pred), clip_values(k2_pred),
        rmse_value = rmse(y_true=f,
                          y_pred=H(kr_pred, phi_pred, a_pred, b_pred, k1_pred, k2_pred, z_train).numpy())

        rmse_dB.append(10 * np.log10(rmse_value))

    with open('./%s/rmse_robust_%s_%f.csv' % (name, parameter_to_robust, robust_index), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in rmse_dB:
            writer.writerow([item])


def clip_values(values, min_value=0, max_value=1):
    return np.clip(values, min_value, max_value)


if __name__ == '__main__':
    # for root, dirs, files in os.walk("."):
    #     for dir_name in dirs:
    #         parameters = process_parameters(dir_name)
    #         subfolder_path = os.path.join(root, dir_name)
    #         output_file_path = os.path.join(subfolder_path, 'output.txt')
    #         csv_output_path = os.path.join(subfolder_path, 'rmse_data.csv')

    names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
             "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
    robust_index = 0.01
    for parameter_to_robust in ("kr", "phi", "a", "b", "k1", "k2"):

        for name in names:
            if name == names[0]:
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ReLU(beta=beta)
                    robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust, name=name + str(beta))
            if name == names[1]:
                for alpha in (0.6, 0.8, 1.0):
                    for beta in (0.55, 0.65, 0.75, 0.85):
                        f = Cliped_ReLU(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[2]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = ELu(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[3]:
                for alpha in (6., 8., 10.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = GeLu(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[4]:
                for alpha in (0.1, 0.15, 0.2):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = Parametric_ReLu(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[5]:
                for alpha in (10., 15., 20.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = SiLU(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[6]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.5, 0.6, 0.7):
                        f = Gaussian(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[7]:
                f = Quadratic(alpha=alpha, beta=beta)
                robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust, name=name)
            if name == names[8]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Sigmoid(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[9]:
                for alpha in (1., pi, 2 * pi):
                    beta = (- pi / 2)
                    f = Sine(alpha=alpha, beta=beta)
                    robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                           name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[10]:
                for alpha in (10., 15., 20.):
                    for beta in (0.2, 0.3, 0.4):
                        f = Softplus(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[11]:
                for alpha in (5., 10., 15.):
                    for beta in (0.5, 0.7):
                        f = Tanh(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[12]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Softsign(alpha=alpha, beta=beta)
                        robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                               name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[13]:
                for beta in (3., 6., 9.):
                    f = Exponential(beta=beta)
                    robust(f, robust_index=robust_index, parameter_to_robust=parameter_to_robust,
                           name=name + "_beta=" + str(beta))
