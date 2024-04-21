# This is k1 sample Python script.
import numpy as np
import matplotlib.pyplot as plt

class OANN():
    def __init__(self, lam):
        self.lam = lam

    def transfer_function_ORR(self, k: float = None, gamma: float = None, neff: float = None,
                              phi: float = None, L: float = None):
        c = np.sqrt(1 - k)
        beta = 2 * np.pi * neff / self.lam
        x = np.exp(-1j * beta * L)
        return (c - gamma * x * np.exp(-1j * phi)) / (1 - c * gamma * x * np.exp(-1j * phi))

    def transfer_function_MZI(self, k1: float = None, k2: float = None, phi: list = []):
        c1 = np.sqrt(1 - k1)
        c2 = np.sqrt(1 - k2)
        s1 = np.sqrt(k1)
        s2 = np.sqrt(k2)

        return -1j * (s2 * c1 * np.exp(-1j * phi) + c2 * s1)

    def transfer_function_MZI_ORR(self, HR, k1: float = None, k2: float = None,
                                  phi: float = None):
        c1 = np.sqrt(1 - k1)
        c2 = np.sqrt(1 - k2)
        s1 = np.sqrt(k1)
        s2 = np.sqrt(k2)

        return -1j * (s2 * c1 * HR * np.exp(-1j * phi) + c2 * s1)

    def calculate_phi(self, a: float = None, b: float = None, z: list = []):
        return a + b * z ** 2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    oann = OANN(lam=1.55)
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / 100)
    function = np.empty([100], dtype=complex)
    for i in range(100):
        function[i] = abs(oann.transfer_function_MZI(k1=0.5, k2=0.3, phi=phi[i])) ** 2
    plt.figure()
    plt.plot(phi, function, linestyle='-')
    # plt.ylabel('%s%s' % (field, field_axis))
    # plt.xlabel("timesteps")
    # plt.title("%s%s-t" % (field, field_axis))
    # file_name = "%s%s" % (field, field_axis)
    # plt.savefig(os.path.join(folder, f"{file_name}.png"))
    # plt.close()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
