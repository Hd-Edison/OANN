# This is k1 sample Python script.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class OANN():
    global pi
    pi = tf.constant(np.pi)


    def __init__(self, lam):
        # unit in m
        self.lam = lam

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


# Press the green button in the gutter to run the script.
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
