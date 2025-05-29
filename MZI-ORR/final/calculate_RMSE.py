import numpy as np
import tensorflow as tf
from main import OANN
from activation_functions import *
import matplotlib.pyplot as plt
import pickle
import csv
import os
from tensorflow.keras.losses import Loss

pi = np.pi
oann = OANN(lam=1550e-9)

import re

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
        rmse_match = rmse_pattern.search(content)
        if param_match:
            parameters['k1'] = float(param_match.group(1))
            parameters['k2'] = float(param_match.group(2))
            parameters['kr1'] = float(param_match.group(3))
            parameters['kr2'] = float(param_match.group(4))
            parameters['phi'] = float(param_match.group(5))
            parameters['a'] = float(param_match.group(6))
            parameters['b'] = float(param_match.group(7))
            parameters['phir2'] = float(param_match.group(8))
        if rmse_match:
            rmse_values = float(rmse_match.group(1))

    return parameters, np.array(rmse_values)


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters


def H(k1, k2, kr1, kr2, phi, a, b, phir2, z_squared=None, scale_to_SI=True):
    if z_squared is None:
        z_squared = tf.range(0, 1, 1 / 200)
    if scale_to_SI:
        phi *= 2 * pi
        a *= 2 * pi
        b *= 10
        phir2 *= 2 * pi
    return abs(oann.transfer_function_MZI_2MRR(HR1=oann.transfer_function_MRR(k=kr1, gamma=0.86, neff=2.389,
                                                                              phi=oann.calculate_phi(a,
                                                                                                     b,
                                                                                                     z_squared=z_squared),
                                                                              L=100e-6 * 2 * np.pi),
                                               HR2=oann.transfer_function_MRR(k=kr2, gamma=0.86, neff=2.389,
                                                                              phi=phir2,
                                                                              L=100e-6 * 2 * np.pi), k1=k1, k2=k2,
                                               phi=phi)) ** 2 * z_squared


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def plot_loss(loss, name=None):
    plt.plot(loss, linewidth=2, linestyle='--')
    plt.title('The loss of training Model', fontname='Times New Roman', fontsize=18, loc='center')
    plt.xlabel('epochs', fontname='Times New Roman', fontsize=18)
    plt.ylabel('loss', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend(['train'])
    plt.savefig("./%s/loss.png" % name)
    # plt.show()
    plt.close()


def plot_prediction_curve(z_squared, function, predicted_function, name):
    plt.plot(z_squared, function, linewidth=2, label="True")
    plt.plot(z_squared, predicted_function, linewidth=2, linestyle='--', label="Prediction")
    # plt.title(name, fontname='Times New Roman', fontsize=18, loc='center')
    plt.xlabel('z^2', fontname='Times New Roman', fontsize=18)
    plt.ylabel('f(z^2)', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend()
    plt.savefig('./%s/prediction.png' % name)
    # plt.show()
    plt.close()


def calculate(true_function, name, z_squared=None):
    if not z_squared:
        z_squared = tf.range(0, 1, 1 / 100)

    parameters = process_parameters(f"./{name}")
    kr1 = parameters["kr1"]
    kr2 = parameters["kr2"]
    phi = parameters["phi"]
    phir2 = parameters["phir2"]
    a = parameters["a"]
    b = parameters["b"]
    k1 = parameters["k1"]
    k2 = parameters["k2"]
    parameters = np.array([k1, k2, kr1, kr2, phi, a, b, phir2])

    current_directory = os.path.dirname(__file__)
    target_folder = os.path.join(current_directory, name)
    os.makedirs(target_folder, exist_ok=True)

    def objective_function(params, y_true):
        # 解包CMA-ES传递的参数
        k1, k2, kr1, kr2, phi, a, b, phir = params

        # 调用H函数
        y_pred = H(k1, k2, kr1, kr2, phi, a, b, phir, z_squared,scale_to_SI=False).numpy()

        # 计算误差，例如最小化均方误差
        rmse = np.sqrt(np.mean(np.square(y_true.numpy() - y_pred)))

        return rmse

    # 输出结果
    f = true_function.function(z_squared)

    rmse_dB = 10 * np.log10(objective_function(parameters, f))
    with open('./%s/output_100.txt' % name, 'w') as f:
        f.write("k1: %f,\n k2: %f,\n kr1: %f,\n kr2: %f,\n phi: %f,\n a: %f,\n b: %f,\n phir2: %f,\n" % (
            k1[0], k2[0], kr1[0], kr2[0], phi[0] * 2 * np.pi, a[0] * 2 * np.pi, b[0] * 10, phir2[0] * 2 * np.pi))
        f.write("rmse = " + str(rmse_dB) + " dB")


if __name__ == '__main__':
    names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
             "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
    for name in names:
        if name == names[0]:
            for beta in (0.15, 0.25, 0.35, 0.55):
                f = ReLU(beta=beta)
                calculate(f, name=name + str(beta))
        if name == names[1]:
            for alpha in (0.6, 0.8, 1.0):
                for beta in (0.55, 0.65, 0.75, 0.85):
                    f = Cliped_ReLU(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[2]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ELu(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[3]:
            for alpha in (6., 8., 10.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = GeLu(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[4]:
            for alpha in (0.1, 0.15, 0.2):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = Parametric_ReLu(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[5]:
            for alpha in (10., 15., 20.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = SiLU(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[6]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.5, 0.6, 0.7):
                    f = Gaussian(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[7]:
            f = Quadratic()
            calculate(f, name=name)
        if name == names[8]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Sigmoid(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[9]:
            for alpha in (1., np.pi, 2 * np.pi):
                beta = (- np.pi / 2)
                f = Sine(alpha=alpha, beta=beta)
                calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[10]:
            for alpha in (10., 15., 20.):
                for beta in (0.2, 0.3, 0.4):
                    f = Softplus(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[11]:
            for alpha in (5., 10., 15.):
                for beta in (0.5, 0.7):
                    f = Tanh(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[12]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Softsign(alpha=alpha, beta=beta)
                    calculate(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[13]:
            for beta in (3., 6., 9.):
                f = Exponential(beta=beta)
                calculate(f, name=name + "_beta=" + str(beta))
import subprocess

subprocess.run(["python", "save_result_to_csv.py"])
