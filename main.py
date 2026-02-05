import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def neff(lam):
    # calculate effective index based on wavelength
    # 500nm * 220nm Si waveguide, substrate = cover = air
    return -1.3e6 * lam + 4.35  # Example linear dependence

class OANN():

    global pi
    pi = tf.constant(np.pi)
    def __init__(self, lam):
        # unit in m
        self.lam = lam

    def transfer_matrix_DC(self, kc: float = None, L: float = None, gamma_0: float = None):

        T = np.zeros([2, 2], dtype=np.complex64)
        T[0] = np.cos(kc * L), -1j * np.sin(kc * L)
        T[1] = -1j * np.sin(kc * L), np.cos(kc * L)
        T *= np.exp(-1j * gamma_0 * L)

        # Convert the numpy array to a TensorFlow tensor
        return tf.cast(T, tf.complex64)

    def transfer_function_ADMRR_through(self,tao_1: float = None, tao_2: float = None,
                                a_rt: float = None, phi_rt: tf.Tensor or list = None):

        H = tao_1-tao_2*a_rt* tf.exp(-1j * phi_rt) / (1-tao_1*tao_2*a_rt* tf.exp(-1j * phi_rt))

        return tf.cast(H, tf.complex64)
    def transfer_function_MRR(self, k: float = None, gamma: float = None, neff: float = None,
                              phi: float = None, L: float = None):

        c = tf.cast(tf.sqrt(1 - k), tf.complex64)
        beta = tf.cast(2 * pi * neff / self.lam, tf.complex64)
        x = tf.cast(tf.exp(-1j * beta * L), tf.complex64)
        gamma = tf.cast(gamma, tf.complex64)

        is_complex = (isinstance(phi, tf.Tensor) and phi.dtype.is_complex) or \
                     (isinstance(phi, np.ndarray) and np.issubdtype(phi.dtype, np.complexfloating))
        if is_complex:
            phi = tf.cast(phi, tf.float32)

        phi_complex = tf.cast(tf.complex(phi, tf.zeros_like(phi)), tf.complex64)

        return (c - gamma * x * tf.exp(-1j * phi_complex)) / (1 - c * gamma * x * tf.exp(-1j * phi_complex))

    def transfer_function_MZI(self, k1: float = None, k2: float = None, phi: tf.Tensor or list = None, HR1=1,
                              HR2=1) -> tf.complex64:
        # Here is why k must < 1 and > 0
        c1 = tf.cast(tf.sqrt(1 - k1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - k2), tf.complex64)
        s1 = tf.cast(tf.sqrt(k1), tf.complex64)
        s2 = tf.cast(tf.sqrt(k2), tf.complex64)

        phi_complex = tf.complex(phi, tf.zeros_like(phi))

        return -1j * (s2 * c1 * tf.exp(-1j * phi_complex) * HR1 + c2 * s1 * HR2)

    def transfer_function_MRR_drop(self, kr1: float = None, kr2: float = None, phi_r: tf.Tensor or list = None,
                                   neff=2.389,
                                   gamma: float = 0.86, r=100e-6, l=500e-6, phi=0) -> tf.complex64:

        c1 = tf.cast(tf.sqrt(1 - kr1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - kr2), tf.complex64)
        s1 = tf.cast(tf.sqrt(kr1), tf.complex64)
        s2 = tf.cast(tf.sqrt(kr2), tf.complex64)
        beta = tf.cast(2 * pi * neff / self.lam, tf.complex64)
        x = tf.cast(tf.exp(-1j * beta * l), tf.complex64)
        gamma = tf.cast(gamma, tf.complex64)

        phi_r_complex, phi_complex = tf.complex(phi_r, tf.zeros_like(phi_r)), tf.complex(phi, tf.zeros_like(phi))
        phi_r_complex, phi_complex = tf.cast(phi_r_complex, tf.complex64), tf.cast(phi_complex, tf.complex64)
        cycle_length = tf.cast(r * 2 * pi, tf.complex64)
        m = gamma * x * tf.exp(-1j * (beta * cycle_length + phi_r_complex))
        return -s1 * s2 * tf.sqrt(m) * tf.exp(-1j * (beta * l + phi_complex)) / (1 - c1 * c2 * m)

    def transfer_function_MZI_MRR(self, HR=1, k1: float = None, k2: float = None,
                                  phi: float = None):
        # for k in (k1, k2):
        #     if tf.rank(k) > 0:
        #         print(tf.rank(k))
        #         tf.debugging.assert_greater_equal(tf.reduce_min(k1), 0., message="Error: k1 contains values less than 0!")
        #         tf.debugging.assert_less_equal(tf.reduce_max(k1), 1., message="Error: k1 contains values larger than 1!")

        c1 = tf.cast(tf.sqrt(1 - k1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - k2), tf.complex64)
        s1 = tf.cast(tf.sqrt(k1), tf.complex64)
        s2 = tf.cast(tf.sqrt(k2), tf.complex64)

        phi_complex = tf.cast(tf.complex(phi, tf.zeros_like(phi)), tf.complex64)

        return -1j * (s2 * c1 * HR * tf.exp(-1j * phi_complex) + c2 * s1)
    def transfer_function_MZI_2ORR(self, HR1, HR2, k1: float = None, k2: float = None,
                                  phi: float = None):
        # Here is why k must < 1 and > 0
        # if tf.cond(k1 < 0) or k1 > 1 or k2 < 0 or k2 > 1:
        #     raise ValueError("k must < 1 and > 0, k1 = %f, k2 = %f" % (k1, k2))

        def raise_error():
            # This function can be used to assert the condition and print an error message
            tf.print("k must < 1 and > 0, k1 =", k1, ", k2 =", k2)
            # To stop execution, you could use an invalid operation
            return tf.constant("k must < 1 and > 0")

        # Define the main computation or error handling based on condition
        result = tf.cond(
            tf.reduce_all([k1 >= 0, k1 <= 1, k2 >= 0, k2 <= 1]),
            lambda: tf.constant("k1 and k2 are in range"),
            raise_error
        )

        # Run the TensorFlow computation
        tf.print(result)

        c1 = tf.cast(tf.sqrt(1 - k1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - k2), tf.complex64)
        s1 = tf.cast(tf.sqrt(k1), tf.complex64)
        s2 = tf.cast(tf.sqrt(k2), tf.complex64)

        phi_complex = tf.cast(tf.complex(phi, tf.zeros_like(phi)), tf.complex64)

        return -1j * (s2 * c1 * HR1 * tf.exp(-1j * phi_complex) + c2 * s1 * HR2)

    def transfer_function_MZI_2MRR(self, HR1, HR2, k1: float = None, k2: float = None,
                                   phi: float = None):
        c1 = tf.cast(tf.sqrt(1 - k1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - k2), tf.complex64)
        s1 = tf.cast(tf.sqrt(k1), tf.complex64)
        s2 = tf.cast(tf.sqrt(k2), tf.complex64)

        phi_complex = tf.cast(tf.complex(phi, tf.zeros_like(phi)), tf.complex64)

        return -1j * (s2 * c1 * HR1 * tf.exp(-1j * phi_complex) + c2 * s1 * HR2)

    def calculate_phi(self, a: float = None, b: float = None, z: list = [], z_squared: list = None):
        if z_squared is None:
            z_squared = tf.square(z)
        return a + b * z_squared

    def calculate_k(self, a: float = None, b: float = None, z: list = [], z_squared: list = None):
        # suppose k = sin(a+b*z^2)^2
        if z_squared is None:
            z_squared = tf.square(z)
        result = tf.square(tf.sin(a + b * z_squared))

        # 限制返回值在0和1之间
        result_clipped = tf.clip_by_value(result, 0.0, 1.0)
        return result_clipped * 0.99 + 0.001

    def k(self, k0, phi, l):
        delta_n=self.lam*phi/(2*pi*l)
        delta_beta = 2*pi*delta_n/self.lam
        S = tf.sqrt(k0+delta_beta**2/4)
        return k0*tf.sin(S*l)**2/(S**2)

    def transfer_function_PD_g1(self, alpha, phi0, I_pi, Pcin, Rpd, Psin, delta_phi=None):
        if delta_phi is None:
            delta_phi = Pcin * Rpd * pi / I_pi
        return alpha ** 2 * tf.sin((phi0 / 2 + delta_phi / 2)) ** 2 * Psin

    def transfer_function_PD_g2(self, alpha, t1, t2, I_pi, Pcin, Rpd, Psin):
        son = t1 ** 2 + (alpha * t2) ** 2 - 2 * alpha * t1 * t2 * tf.cos(Pcin * Rpd * pi / I_pi)
        mom = 1 + (alpha * t1 * t2) ** 2 - 2 * alpha * t1 * t2 * tf.cos(Pcin * Rpd * pi / I_pi)
        return son * Psin / mom

    def calculate_a_b(self, alpha=0.1, responsivity=0.8, area=1.0,
                      V_pi=10.0, V_bias=10.0, R=1e3, impedance=120 * pi, ):
        b = pi * alpha * R * responsivity * area * 1e-12 / 2 / V_pi / impedance
        a = pi * V_bias / V_pi
        return a, b

class DC():
    def __init__(self, Kappa=None, tau=None, kc=None, L=None, gamma_0=None):
        """
        Directional Coupler (DC) model.
        :param Kappa: field coupling coefficient, Kappa_DC = sin(kc * L)
        :param tau: field transmission coefficient, for lossless coupler, tau**2 + Kappa_DC**2 = 1, tau = cos(kc * L)
        :param kc: Coupling strength um^-1
        :param L: Coupling Length um
        :param gamma_0: Average propagation constant
        """
        if Kappa is not None:
            self.Kappa = Kappa
        elif kc is not None:
            self.Kappa = np.sin(kc * L)
        if tau is not None:
            self.tau = tau
        elif kc is not None:
            self.tau = np.cos(kc * L)
        else:
            self.tau = np.sqrt(1 - self.Kappa ** 2)
        self.kc = kc
        self.L = L
        self.gamma_0 = gamma_0

    def transfer_matrix(self):
        """
        Transfer matrix for a directional coupler (DC).
        see p41, eq(1.131) of Optical Microring Resonators - V.Van
        :return: Transfer matrix
        """
        T = np.zeros([2, 2], dtype=np.complex64)
        T[0] = self.tau, -1j * self.Kappa
        T[1] = -1j * self.Kappa, self.tau
        phase = np.exp(-1j * self.gamma_0 * self.L)  # phase factor
        # phase shape = [100, 1]
        phase = np.atleast_1d(phase)# 保证phase至少是一维
        T = T[None, :, :] * phase[:, None, None]  # shape = [100, 2, 2]
        T = np.squeeze(T, axis=1) if T.shape[1] == 1 else T  # 如果有多余的1维，去掉

        return tf.cast(T, dtype=tf.complex64)

class Ring():
    def __init__(self, tau_1: float = None, tau_2: float = None, a_rt: float = None, phi_rt: list or float = None):
        """
        add-drop microring resonator (ADMRR)
        :param tau_1: transmission coefficient 1
        :param tau_2: transmission coefficient 2
        :param a_rt: roundtrip loss, can be a list or a single value
        :param phi_rt: roundtrip phase shift, can be a list or a single value
        """
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.a_rt = a_rt
        self.phi_rt = phi_rt

    def transfer_function_through(self):
        """
        Transfer function for an add-drop microring resonator (ADMRR).
        see p61, eq(2.24) of Optical Microring Resonators - V.Van
        For All-pass transfer function, set tau_2 = 1.
        :return: Transfer function
        """

        exp_arg = tf.complex(tf.zeros_like(self.phi_rt), -self.phi_rt)
        H = ((self.tau_1 - self.tau_2 * self.a_rt * tf.exp(exp_arg)) /
             (1 - self.tau_1 * self.tau_2 * self.a_rt * tf.exp(exp_arg)))


        return tf.cast(H, dtype=tf.complex64)

class RMZI():
    def __init__(self, DC1: DC, DC2: DC, Ring, gamma_0, L_MZI, delta_phi, alpha: float = 0):
        """
        Ring assisted Mach-Zehnder Interferometer (RMZI).
        :param DC1: First directional coupler
        :param DC2: Second directional coupler
        :param Ring: Ring resonator
        :param gamma_0: Average propagation constant
        :param L_MZI: Length of the Mach-Zehnder Interferometer arms in meters
        :param delta_phi: Phase difference between the two arms of the MZI
        :param alpha: field Loss coefficient in 1/m
        """
        self.T_DC1 = DC1.transfer_matrix()
        self.T_DC2 = DC2.transfer_matrix()
        self.H_ring = Ring.transfer_function_through()
        self.gamma_0 = gamma_0
        self.L_MZI = L_MZI
        self.delta_phi = delta_phi
        self.alpha = alpha

    def transfer_matrix(self):
        """
        TODO: 直波导和DC的loss？
        Transfer matrix for a ring assisted Mach-Zehnder Interferometer (RMZI).
        :return: transfer matrix of RMZI
        """
        H_ring_np = self.H_ring.numpy()  # Convert the TensorFlow tensor to a NumPy array
        # TODO: exp里面虚部的符号是正还是负？我认为都是负，因为是正向传播。
        L_arm1 = np.exp(-1 * self.alpha * self.L_MZI - 1j * self.gamma_0 * self.L_MZI) * H_ring_np
        L_arm2 = np.exp(-1 * self.alpha * self.L_MZI - 1j * (self.gamma_0 * self.L_MZI + self.delta_phi))
        # 先变成 shape [100,]
        L_arm1 = np.squeeze(L_arm1)
        L_arm2 = np.squeeze(L_arm2)

        # 构造 [100, 2, 2] 的 batch 对角阵
        T_arms = np.stack([
            np.stack([L_arm1, np.zeros_like(L_arm1)], axis=-1),
            np.stack([np.zeros_like(L_arm2), L_arm2], axis=-1)
        ], axis=1, dtype=complex)  # shape [100, 2, 2]
        T_arms_tf = tf.convert_to_tensor(T_arms, dtype=tf.complex64)

        # 计算 T_DC2 * T_arms * T_DC1
        return tf.matmul(self.T_DC2, tf.matmul(T_arms_tf, self.T_DC1))

class MVM():
    def __init__(self, scale: int=None, RMZIs: list=None):
        """
        Matrix-vector multiplication (MVM) model.
        :param scale: Scale factor for the MVM
        :param RMZIs: List of RMZI objects, it's a 2D list with a shape of [scale, scale]
        :type scale: int
        :type RMZIs: list of RMZI
        """
        self.scale = scale
        self.RMZIs = RMZIs

    def calculate_output(self, input: tf.Tensor) -> tf.Tensor:
        """
        Calculate the output of the MVM.
        :param input: Input value to the MVM
        :return: Output of the MVM with shape [scale, scale, 2]
        """
        # 使用列表构建输出，避免tensor赋值问题
        outputs_list = []

        # 初始化输入矩阵
        input_val = tf.cast(input, tf.complex64)
        current_inputs = tf.zeros([self.scale, self.scale, 2], dtype=tf.complex64)

        # 设置第一行的输入
        first_row_inputs = []
        for j in range(self.scale):
            if j < 3:  # 前三个元素设置为input
                first_row_inputs.append([input_val, tf.cast(0.0, tf.complex64)])
            else:
                first_row_inputs.append([tf.cast(0.0, tf.complex64), tf.cast(0.0, tf.complex64)])

        # 构建完整的输入tensor
        inputs_data = []
        for i in range(self.scale):
            if i == 0:
                inputs_data.append(first_row_inputs)
            else:
                inputs_data.append(
                    [[tf.cast(0.0, tf.complex64), tf.cast(0.0, tf.complex64)] for _ in range(self.scale)])

        current_inputs = tf.convert_to_tensor(inputs_data, dtype=tf.complex64)
        # current_inputs是对的
        # print(current_inputs.numpy())
        # print(current_inputs[0, 0, 0].numpy())
        # print(current_inputs[0, 1, 0].numpy())
        # print(current_inputs[0, 2, 0].numpy())
        # print(current_inputs[0, 0, 1].numpy())
        # print(current_inputs[0, 1, 1].numpy())
        # print(current_inputs[0, 2, 1].numpy())
        # print(current_inputs[1, 0, 0].numpy())
        # print(current_inputs[1, 1, 0].numpy())
        # print(current_inputs[2, 0, 0].numpy())
        # print(current_inputs[2, 1, 0].numpy())
        # 把current_inputs的形状调整为[1000, scale, scale, 2]，以便进行批处理
        current_inputs = tf.tile(current_inputs[tf.newaxis, :, :, :], [1000, 1, 1, 1])

        # 逐行处理
        for i in range(self.scale):
            row_outputs = []
            for j in range(self.scale):
                # 获取当前输入并计算输出
                print(f"Calculating RMZI_{i+1}{j+1}")
                input_vector = current_inputs[:, i, j, :]
                input_vector = tf.reshape(input_vector, [1000, 2, 1])
                print(f"RMZI_{i+1}{j+1}",input_vector[0, :, 0])
                """可能与计算顺序有关，上一个rmzi的输出未计算前不能先计算下一个RMZI"""
                """RMZI12和13的输入有问题"""
                # 计算RMZI输出
                out_tem = tf.matmul(self.RMZIs[i][j].transfer_matrix(), input_vector)
                output_flat = tf.squeeze(out_tem)
                row_outputs.append(output_flat)

                # 更新下一行和下一列的输入（使用scatter_nd_update）
                if i < self.scale - 1:
                    # 构建批量索引：[1000, 4] - [batch_idx, i+1, j, 0]
                    batch_indices = tf.range(1000)[:, tf.newaxis]  # [1000, 1]
                    spatial_indices = tf.constant([[i + 1, j, 0]])  # [1, 3]
                    indices = tf.concat([batch_indices, tf.tile(spatial_indices, [1000, 1])], axis=1)  # [1000, 4]
                    updates = output_flat[:, 0]  # [1000]
                    current_inputs = tf.tensor_scatter_nd_update(current_inputs, indices, updates)

                if j < self.scale - 1:
                    # 构建批量索引：[1000, 4] - [batch_idx, i, j+1, 0]
                    batch_indices = tf.range(1000)[:, tf.newaxis]  # [1000, 1]
                    spatial_indices = tf.constant([[i, j + 1, 1]])  # [1, 3]
                    indices = tf.concat([batch_indices, tf.tile(spatial_indices, [1000, 1])], axis=1)  # [1000, 4]
                    updates = output_flat[:, 1]  # [1000]
                    current_inputs = tf.tensor_scatter_nd_update(current_inputs, indices, updates)
                print(current_inputs.numpy()[0, :, :])
            outputs_list.append(tf.stack(row_outputs))

        Outputs = tf.stack(outputs_list)
        return Outputs





if __name__ == '__main__':
    oann = OANN(lam=1550e-9)
    phi = tf.range(0, 2 * pi, 2 * pi / 100)
    z = tf.range(0, 1, 1 / 100)
    function = np.empty([100], dtype=complex)
    kr = 0.5
    gamma = 1.
    neff = 2.389
    phi_r = 0.
    L = 100e-6
    a, b = oann.calculate_a_b()
    print(a, b)
    for i in range(100):
        function[i] = abs(
            oann.transfer_function_MZI_ORR(HR=oann.transfer_function_MRR(k=kr, gamma=gamma, neff=neff, phi=phi_r, L=L),
                                      k1=0.717, k2=0.8, phi=0.))
    plt.figure()
    plt.plot(z ** 2, (function * z) ** 2, linestyle='-')
    # plt.ylabel('%s%s' % (field, field_axis))
    # plt.xlabel("timesteps")
    # plt.title("%s%s-t" % (field, field_axis))
    # file_name = "%s%s" % (field, field_axis)
    # plt.savefig(os.path.join(folder, f"{file_name}.png"))
    # plt.close()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
