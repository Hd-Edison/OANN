# This is k1 sample Python script.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class OANN():

    global pi
    pi = tf.constant(np.pi)

    def __init__(self, lam):
        self.lam = lam

    def transfer_function_ORR(self, k: float = None, gamma: float = None, neff: float = None,
                              phi: float = None, L: float = None):
        c = tf.sqrt(1 - k)
        beta = 2 * pi * neff / self.lam
        x = tf.exp(-1j * beta * L)
        return (c - gamma * x * tf.exp(-1j * phi)) / (1 - c * gamma * x * tf.exp(-1j * phi))

    def transfer_function_MZI(self, k1: float = None, k2: float = None, phi: tf.Tensor or list = None) -> tf.complex64:
        # Here is why k must < 1 and > 0
        if k1 < 0 or k1 > 1 or k2 < 0 or k2 > 1:
            raise ValueError("k must < 1 and > 0, k1 = %f, k2 = %f" % (k1, k2))
        c1 = tf.cast(tf.sqrt(1 - k1), tf.complex64)
        c2 = tf.cast(tf.sqrt(1 - k2), tf.complex64)
        s1 = tf.cast(tf.sqrt(k1), tf.complex64)
        s2 = tf.cast(tf.sqrt(k2), tf.complex64)

        phi_complex = tf.complex(phi, tf.zeros_like(phi))

        return -1j * (s2 * c1 * tf.exp(-1j * phi_complex) + c2 * s1)

    def transfer_function_MZI_ORR(self, HR, k1: float = None, k2: float = None,
                                  phi: float = None):
        c1 = tf.sqrt(1 - k1)
        c2 = tf.sqrt(1 - k2)
        s1 = tf.sqrt(k1)
        s2 = tf.sqrt(k2)

        return -1j * (s2 * c1 * HR * tf.exp(-1j * phi) + c2 * s1)

    def calculate_phi(self, a: float = None, b: float = None, z: list = [], z_squared: list = None):
        if z_squared is None:
            z_squared = tf.square(z)
        return a + b * z_squared


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    oann = OANN(lam=1.55)
    phi = tf.range(0, 2 * np.pi, 2 * np.pi / 100)
    z = tf.range(0, 1, 1 / 100)
    function = np.empty([100], dtype=complex)
    for i in range(100):
        function[i] = abs(oann.transfer_function_MZI(k1=0.5, k2=0.3, phi=phi[i])) ** 2
    plt.figure()
    plt.plot(phi, (function * z) ** 2, linestyle='-')
    # plt.ylabel('%s%s' % (field, field_axis))
    # plt.xlabel("timesteps")
    # plt.title("%s%s-t" % (field, field_axis))
    # file_name = "%s%s" % (field, field_axis)
    # plt.savefig(os.path.join(folder, f"{file_name}.png"))
    # plt.close()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
