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


k1_values = tf.range(0, 1, 1/30)
k2_values = k1_values
a_values = k1_values
# a_values = tf.range(0, 2 * np.pi, np.pi / 15)
b_values = k1_values

phi_values = tf.range(0, 2 * np.pi, np.pi / 15)

z_squared = tf.range(0, 1, 1 / 200)
oann = OANN(lam=1.55)
# phi_values = oann.calculate_phi(a)
# 生成所有参数排列组合
parameters = []
for k1 in k1_values:
    for k2 in k2_values:
        for a in a_values:
            for b in b_values:
                result = abs(oann.transfer_function_MZI_ORR(HR=oann.transfer_function_ORR(),k1=k1, k2=k2, phi=oann.calculate_phi(a * 2 * np.pi, b, z_squared=z_squared))) ** 2 * z_squared
                parameters.append([k1._numpy(), k2._numpy(), a._numpy(), b._numpy()] + list(result._numpy()))

# parameters = [np.float16(x) for x in parameters]

# 转换为DataFrame
df = pd.DataFrame(parameters)

# 保存为CSV文件
df.to_csv('RELU_MZI_training_data.csv', index=False, header=False)
