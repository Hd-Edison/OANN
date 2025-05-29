import numpy as np
import tensorflow as tf
from main import OANN
from activation_functions import *
import os
import re
from tensorflow.keras.losses import Loss
import pandas as pd
import INVERSE_MRR.load_model as INVERSE_MODEL_MRR
import INVERSE_DC.load_model as INVERSE_MODEL_DC

pi = np.pi

oann = OANN(lam=1550e-9)


class CustomLoss(Loss):
    def __init__(self, true_function):
        self.true_function = true_function
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        kr, phi, a, b, k1, k2 = tf.split(y_pred, num_or_size_splits=6, axis=1)
        # tf.gather 不能大于batch size
        kr, phi, a, b, k1, k2 = tf.gather(kr, [0]), tf.gather(phi, [0]), tf.gather(a, [0]), tf.gather(b, [0]), \
            tf.gather(k1, [0]), tf.gather(k2, [0])
        temp = self.true_function.function()
        H_pred = H(kr, phi, a, b, k1, k2)
        return tf.reduce_mean(tf.square(temp - H_pred))


def H(kr, phi, a, b, k1, k2, z_squared=None):
    if z_squared is None:
        z_squared = tf.range(0, 1, 1 / 200)

    return abs(oann.transfer_function_MZI_MRR(HR=oann.transfer_function_MRR(k=kr, gamma=0.86, neff=2.389,
                                                                            phi=oann.calculate_phi(a=a * 2 * pi,
                                                                                                   b=b * 10,
                                                                                                   z_squared=z_squared),
                                                                            L=100e-6 * 2 * pi), k1=k1, k2=k2,
                                              phi=phi * 2 * pi)) ** 2 * z_squared


def process_parameters(folder_path):
    output_file_path = os.path.join(folder_path, 'output.txt')

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

    if os.path.isfile(output_file_path):
        parameters, rmse_values = extract_parameters(output_file_path)
        return parameters

def load():
    parent_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MRR_data_path = os.path.join(parent_folder, 'INVERSE_MRR', 'dataset_Lc_gap.csv')
    DC_data_path = os.path.join(parent_folder, 'INVERSE_DC', 'DC_dataset_Lc_gap.csv')
    MRR_data = pd.read_csv(MRR_data_path, header=None).iloc[:, 2:]
    DC_data = pd.read_csv(DC_data_path, header=None).iloc[:, 2:]
    MRR_k_at_1550 = MRR_data.iloc[:, 9]
    DC_k_at_1550 = DC_data.iloc[:, 9]
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            parameters = process_parameters(dir_name)
            if parameters is not None:
                kr, k1, k2 = np.array([parameters["kr"], parameters["k1"], parameters["k2"]])
                subfolder_path = os.path.join(root, dir_name)

                closest_index_kr = (MRR_k_at_1550 - kr).abs().idxmin()  # 获取最接近的行索引
                closest_index_k1 = (DC_k_at_1550 - k1).abs().idxmin()  # 获取最接近的行索引
                closest_index_k2 = (DC_k_at_1550 - k2).abs().idxmin()  # 获取最接近的行索引

                prediction_kr = INVERSE_MODEL_MRR.load_model(input_value=MRR_data.iloc[closest_index_kr, :])
                prediction_k1 = INVERSE_MODEL_DC.load_model(input_value=DC_data.iloc[closest_index_k1, :])
                prediction_k2 = INVERSE_MODEL_DC.load_model(input_value=DC_data.iloc[closest_index_k2, :])

                with open('%s/inverse_output.txt' % subfolder_path, 'w') as f:
                    f.write(f"gap_kr: {prediction_kr[0, 0] * 150} nm,\n "
                            f"Lc_kr: {prediction_kr[0, 1] * 5000} nm \n"
                            f"Additional phase shift introduced by Lc_kr: {prediction_kr[0, 1] * 5000 * 3.45 * np.pi * 2 / 1550} \n")
                    f.write(f"\n"
                            f"gap_k1: {prediction_k1[0, 0] * 150} nm,\n "
                            f"Lc_k1: {prediction_k1[0, 1] * 5000} nm \n")
                    f.write(f"\n"
                            f"gap_k2: {prediction_k2[0, 0] * 150} nm,\n "
                            f"Lc_k2: {prediction_k2[0, 1] * 5000} nm \n")


if __name__ == '__main__':
    load()
