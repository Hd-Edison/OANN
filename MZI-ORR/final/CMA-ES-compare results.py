from main import OANN
import cma
import numpy as np
import os
from activation_functions import *
import re
import csv

oann = OANN(lam=1550e-9)
pi = np.pi


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
        rmse_match = rmse_pattern.search(content)
        if param_match:
            parameters['kr'] = float(param_match.group(1))
            parameters['phi'] = float(param_match.group(2))
            parameters['a'] = float(param_match.group(3))
            parameters['b'] = float(param_match.group(4))
            parameters['k1'] = float(param_match.group(5))
            parameters['k2'] = float(param_match.group(6))
        if rmse_match:
            rmse_values = float(rmse_match.group(1))

    return parameters, np.array(rmse_values)


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters


def H(kr, phi, a, b, k1, k2, z_squared):
    return abs(oann.transfer_function_MZI_MRR(HR=oann.transfer_function_MRR(k=kr, gamma=0.86, neff=2.389,
                                                                            phi=oann.calculate_phi(a=a * 2 * np.pi,
                                                                                                   b=b * 10,
                                                                                                   z_squared=z_squared),
                                                                            L=100e-6 * 2 * np.pi),
                                              k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared


def cma_es(true_function, name, z_squared=None):
    if not z_squared:
        z_squared = tf.range(0, 1, 1 / 200)
    # 执行函数，处理指定文件夹下的 output.txt 文件
    parameters = process_parameters(f"./{name}")
    kr = parameters["kr"]
    phi = parameters["phi"]
    a = parameters["a"]
    b = parameters["b"]
    k1 = parameters["k1"]
    k2 = parameters["k2"]
    parameters = [kr, phi, a, b, k1, k2]

    current_directory = os.path.dirname(__file__)
    target_folder = os.path.join(current_directory, name)
    os.makedirs(target_folder, exist_ok=True)

    def objective_function(params, z_squared, y_true):
        # 解包CMA-ES传递的参数
        kr, phi, a, b, k1, k2 = params

        # 调用H函数
        y_pred = H(kr, phi, a, b, k1, k2, z_squared).numpy()

        # 计算误差，例如最小化均方误差
        rmse = np.sqrt(np.mean(np.square(y_true.numpy() - y_pred)))

        return rmse

    # 定义参数的范围 (每个参数的上下限)
    lower_bounds = [0.0001, 0, 0, 0, 0.0001, 0.0001]  # 每个参数的下限
    upper_bounds = [0.9999, 2 * np.pi, 2 * np.pi, 10, 0.9999, 0.9999]  # 每个参数的上限
    result = cma.fmin(objective_function, parameters, 0.5,
                      args=(z_squared, true_function.function()),
                      options={'bounds': [lower_bounds, upper_bounds]})

    kr, phi, a, b, k1, k2 = result[0]
    rmse_dB = 10 * np.log10(result[1])
    with open('./%s/CMA-ES.txt' % name, 'w') as f:
        f.write("kr: %f,\n phi: %f,\n a: %f,\n b: %f,\n k1: %f,\n k2: %f \n" % (
            kr, phi, a, b, k1, k2))
        f.write("rmse = " + str(rmse_dB) + " dB")


def compare_result():
    i = 0


def save_to_csv(parent_folder, iterations):
    """
    遍历文件夹中的子文件夹，比较 output.txt 和 CMA-ES.txt 中的 rmse 值，并将结果保存到 CSV 文件
    :param parent_folder: 父文件夹路径，包含多个子文件夹
    """
    result_list = []
    # 遍历所有子文件夹
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)

        # 确保是一个文件夹
        if os.path.isdir(subfolder_path):
            output_file = os.path.join(subfolder_path, 'output.txt')
            cma_es_file = os.path.join(subfolder_path, 'CMA-ES.txt')

            # 检查文件是否存在
            if os.path.exists(output_file) and os.path.exists(cma_es_file):
                _, output_rmse = extract_parameters(output_file)
                _, cma_es_rmse = extract_parameters(cma_es_file)

                if output_rmse is not None and cma_es_rmse is not None:
                    # 将较小的 RMSE 值保存到列表中
                    smaller_rmse = min(output_rmse, cma_es_rmse)
                    flag = output_rmse > cma_es_rmse
                    result_list.append([subfolder, smaller_rmse, output_rmse, cma_es_rmse, flag])

    # 将结果保存为 CSV 文件
    csv_file = os.path.join(parent_folder, f'CMA-ES//rmse_comparison_{iterations}.csv')
    # a = np.vstack([result_list, output_list, cma_es_list])
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Folder', 'RMSE (dB)', "Original RMSE", "CMA-ES RMSE", "CMA-ES better?"])
        writer.writerows(result_list)

    print(f"Results saved to {csv_file}")


if __name__ == '__main__':
    folder_name = "CMA-ES"
    iterations = 5
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
             "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
    for i in range(iterations):
        # ("ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
        # "Sine", "Softplus", "Tanh", "Softsign", "Exponential")
        for name in names:
            if name == names[0]:
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ReLU(beta=beta)
                    cma_es(f, name=name + str(beta))
            if name == names[1]:
                for alpha in (0.6, 0.8, 1.0):
                    for beta in (0.55, 0.65, 0.75, 0.85):
                        f = Cliped_ReLU(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[2]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = ELu(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[3]:
                for alpha in (6., 8., 10.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = GeLu(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[4]:
                for alpha in (0.1, 0.15, 0.2):
                    for beta in (0.15, 0.25, 0.35, 0.55):
                        f = Parametric_ReLu(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[5]:
                for alpha in (10., 15., 20.):
                    for beta in (0.25, 0.35, 0.45, 0.55):
                        f = SiLU(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[6]:
                for alpha in (0.1, 0.2, 0.3):
                    for beta in (0.5, 0.6, 0.7):
                        f = Gaussian(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[7]:
                f = Quadratic(alpha=alpha, beta=beta)
                cma_es(f, name=name)
            if name == names[8]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Sigmoid(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[9]:
                for alpha in (1., pi, 2 * pi):
                    beta = (- pi / 2)
                    f = Sine(alpha=alpha, beta=beta)
                    cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[10]:
                for alpha in (10., 15., 20.):
                    for beta in (0.2, 0.3, 0.4):
                        f = Softplus(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[11]:
                for alpha in (5., 10., 15.):
                    for beta in (0.5, 0.7):
                        f = Tanh(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[12]:
                for alpha in (10., 20.):
                    for beta in (0.5, 0.7):
                        f = Softsign(alpha=alpha, beta=beta)
                        cma_es(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
            if name == names[13]:
                for beta in (3., 6., 9.):
                    f = Exponential(beta=beta)
                    cma_es(f, name=name + "_beta=" + str(beta))
        save_to_csv(".", i)
