from main import OANN
from cmaes import CMA
from cmaes import CMAwM
import numpy as np
import os
from activation_functions import *
import re
import csv
import pandas as pd
import gc


oann = OANN(lam=1550e-9)
pi = np.pi
# "CMA-ES",
folder_names = ["CMA-ES/only CMA"]

def extract_parameters(file_path):
    # 定义正则表达式以匹配所有参数
    pattern = re.compile(
        r'kr: ([\d.e-]+),\s*phi: ([\d.e-]+),\s*a: ([\d.e-]+),\s*b: ([\d.e-]+),\s*k1: ([\d.e-]+),\s*a_k1: ([\d.e-]+),\s*b_k1: ([\d.e-]+),\s*k2: ([\d.e-]+)',
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
            parameters['a_k1'] = float(param_match.group(6))
            parameters['b_k1'] = float(param_match.group(7))
            parameters['k2'] = float(param_match.group(8))
        if rmse_match:
            rmse_values = float(rmse_match.group(1))

    return parameters, np.array(rmse_values)

def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters



def H(kr, phi, a, b, k1, a_k1, b_k1, k2, z_squared=None, scale_to_SI=False):
    if z_squared is None:
        z_squared = tf.range(0, 1, 1 / 200)
    if scale_to_SI:
        a_k1*=(2*pi)
        a*=(2*pi)
        phi*= (2 * pi)
        b_k1*=10
        b*= 10
    # k1 = oann.calculate_k(a_k1 * 2 * np.pi, b_k1, z_squared=z_squared)
    phi_k=oann.calculate_phi(a_k1, b_k1, z_squared=z_squared)
    k1=oann.k(k1, phi_k, 10e-6)
    return abs(oann.transfer_function_MZI_MRR(HR=oann.transfer_function_MRR(k=kr, gamma=0.86, neff=2.389,
                                                                            phi=oann.calculate_phi(a=a,
                                                                                                   b=b,
                                                                                                   z_squared=z_squared),
                                                                            L=100e-6 * 2 * pi),
                                              k1=k1,
                                              k2=k2, phi=phi)) ** 2 * z_squared


def cma_es(true_function, name, z_squared=None):
    if not z_squared:
        z_squared = tf.range(0, 1, 1 / 200)
    # 执行函数，处理指定文件夹下的 output.txt 文件
    parameters = np.array([0.5, pi, pi, 5, 0.5, pi, 5, 0.5])

    current_directory = os.path.dirname(__file__)
    target_folder = os.path.join(current_directory, name)
    os.makedirs(target_folder, exist_ok=True)

    def objective_function(params, y_true):
        # 解包CMA-ES传递的参数
        kr, phi, a, b, k1, a_k1, b_k1, k2 = params

        # 调用H函数
        y_pred = H(kr, phi, a, b, k1, a_k1, b_k1, k2, scale_to_SI=False).numpy()

        # 计算误差，例如最小化均方误差
        rmse = np.sqrt(np.mean(np.square(y_true.numpy() - y_pred)))

        return rmse

    # 定义参数的范围 (每个参数的上下限)
    bounds = np.array([[0.0001, 0.9999], [0, 2 * np.pi], [0, 2 * np.pi], [0, 10], [0.0001, 0.9999], [0, 2 * np.pi], [0, 10], [0.0001, 0.9999]])
    # folder_names = ["CMA-ES", "LRA-CMA-sigma=0.5-2.0-sigma=0.5-2.0", "CMAwM"]

    optimizer = CMA(mean=parameters, sigma=sigma, bounds=bounds)

    # elif folder_name == folder_names[2]:
    #     # dim可能等于参数数量
    #     steps = np.zeros_like(bounds)
    #     steps[:, 1] = np.array([1, 1, 1, 1, 1, 1])
    #     optimizer = CMAwM(mean=parameters, sigma=0.5, bounds=bounds, steps=steps[:, 0])

    for generation in range(generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = objective_function(params=x, y_true=true_function.function())
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}, x3={x[2]}, x4 = {x[3]}, x5={x[4]}, x6 = {x[5]}), x7 = {x[6]}, x8 = {x[7]})")
        optimizer.tell(solutions)

    kr, phi, a, b, k1, a_k1, b_k1, k2 = solutions[-1][0]
    rmse_dB = 10 * np.log10(solutions[-1][1])
    with open(f'./{name}/{"only-cma"}.txt', 'w') as f:
        f.write("kr: %f,\n phi: %f,\n a: %f,\n b: %f,\n k1: %f,\n a_k1: %f,\n b_k1: %f,\n k2: %f \n" % (
            kr, phi, a, b, k1, a_k1, b_k1, k2))
        f.write("rmse = " + str(rmse_dB) + " dB")
    gc.collect()


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
            cma_es_file = os.path.join(subfolder_path, f'{"only-cma"}.txt')

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
    csv_file = os.path.join(parent_folder, f'{"CMA-ES//only CMA"}//rmse_only-cma.csv')
    # a = np.vstack([result_list, output_list, cma_es_list])
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Folder', 'RMSE (dB)', "Original RMSE", "CMA-ES RMSE", "CMA-ES better?"])
        writer.writerows(result_list)

    print(f"Results saved to {csv_file}")


def compare_result(sigma):
    # 读取第一个文件并获取前两列数据
    df = pd.read_csv(f'{"only-cma"}//rmse_comparison_0_sigma={sigma}.csv', skiprows=1, header=None)
    folder_list = df.iloc[:, 0].tolist()
    rmse_list = df.iloc[:, 2].tolist()

    # 初始化数据列表，data 将是一个二维列表，按行存储
    data = [[folder, rmse] for folder, rmse in zip(folder_list, rmse_list)]

    # 追加其他文件中的数据 (第三列及之后)
    for i in range(6):
        try:
            df = pd.read_csv(f'{"only-cma"}//rmse_comparison_{i}_sigma={sigma}.csv', skiprows=1, header=None)
            cma_es_rmse = df.iloc[:, 3].tolist()
            for j in range(len(data)):
                data[j].append(cma_es_rmse[j])  # 将对应的 CMA-ES RMSE 追加到同一行
        except:
            break

    # 将结果保存为 CSV 文件
    csv_file = os.path.join(".", f'{"only-cma"}//result.csv')

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Folder', 'RMSE (dB)', "CMA-ES RMSE 1", "CMA-ES RMSE 2", "CMA-ES RMSE 3", "CMA-ES RMSE 4", "CMA-ES RMSE 5"])
        writer.writerows(data)

    print(f"Results saved to {csv_file}")


if __name__ == '__main__':
    generations = 1500
    if not os.path.exists(folder_names[0]):
        os.makedirs(folder_names[0])

    for folder_name in folder_names:
        for sigma in [0.25]:
            # # 0.25,0 .5, 0.8, 1. , 1.5, 2.
            iterations = 1
            # Check if the folder already exists
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic",
                     "Sigmoid",
                     "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
            for i in range(iterations):
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

