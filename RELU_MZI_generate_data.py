import numpy as np
import pandas as pd
from main import OANN


# 生成参数范围
def RELU(beta, z):
    m = 1 / (1 - beta)
    result = m * (z - beta)
    result[result < 0] = 0
    return result


k1_values = np.arange(0, 1.05, 0.05)
k2_values = np.arange(0, 1.05, 0.05)
# phi_values = np.arange(0, 2 * np.pi + np.pi / 10, np.pi / 10)
# z_values = np.arange(0, 1.05, 0.05)
# b_values = z_values
# a_values = np.arange(0, np.pi + np.pi / 20, np.pi / 20)
# oann = OANN(lam=1.55)
# # phi_values = oann.calculate_phi(a)
# # 生成所有参数排列组合
# parameters = []
# for k1 in k1_values:
#     for k2 in k2_values:
#         for a in a_values:
#             for b in b_values:
#                 result = abs(oann.transfer_function_MZI(k1=k1, k2=k2, phi=oann.calculate_phi(a, b, z))) ** 2
#                 parameters.append([k1, k2, a, b, result])


# 转换为DataFrame
df = pd.DataFrame(parameters, columns=['k1', 'k2', 'phi', 'result'])

# 保存为CSV文件
df.to_csv('RELU_MZI_training_data.csv', index=False)
