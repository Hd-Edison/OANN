import numpy as np
import pandas as pd
from main import OANN
import tensorflow as tf


# 生成参数范围
def RELU(beta, z):
    m = 1 / (1 - beta)
    result = m * (z - beta)
    result[result < 0] = 0
    return result


k1_values = tf.range(0, 1, 1 / 20)
k_values = k1_values
a_values = tf.range(0, 1, 1 / 20)
# b_values = k1_values

phi_values = tf.range(0, 2 * np.pi, np.pi / 10)

z_squared = tf.range(0, 1, 1 / 100)
oann = OANN(lam=1.55)
# phi_values = oann.calculate_phi(a)
# 生成所有参数排列组合
parameters = []
for b in a_values:
    for kr in k_values:
        for a in a_values:
            for phi in k_values:
                for k1 in k_values:
                    for k2 in k_values:
                        result = abs(
                            oann.transfer_function_MZI_ORR(HR=oann.transfer_function_ORR(k=kr, gamma=0.8, neff=2.384556,
                                                                                         phi=oann.calculate_phi(a * 10 * 2 * np.pi, b * 10, z_squared=z_squared), L=5e-6 * 2 * np.pi),
                                                           k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared
                        parameters.append([kr._numpy(), phi._numpy(), a._numpy(), b._numpy(), k1._numpy(), k2._numpy()] + list(result._numpy()))

# parameters = [np.float16(x) for x in parameters]

# 转换为DataFrame
df = pd.DataFrame(parameters)

# 保存为CSV文件
df.to_csv('training_data.csv', index=False, header=False)
