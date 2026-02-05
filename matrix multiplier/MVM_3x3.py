import numpy as np
from main import *
from numpy import pi
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

"""MVM"""


def power(input):
    return np.abs(input) ** 2


def neff(lam):
    # calculate effective index based on wavelength
    # 500nm * 220nm Si waveguide, substrate = cover = air
    return -1.3e6 * lam + 4.35  # Example linear dependence


if __name__ == '__main__':
    """以波长为自变量"""
    # start_lam = 1545e-9  # Start wavelength in meters
    # end_lam = 1555e-9  # End wavelength in meters
    # lam = np.arange(start_lam, end_lam, (end_lam - start_lam) / 100)  # Wavelength in meters
    # lam = lam.reshape(-1, 1)  # Reshape for broadcasting

    """固定波长"""
    lam = 1.5458e-06
    # lam = 1.5481e-06
    # Wavelength in meters
    Kappa_DC = np.sqrt(0.5)  # Coupling strength in um^-1
    L_DC = 11  # Length of the coupling waveguide in um
    n_eff = neff(lam)  # Effective index of the waveguide neff(lam=1.7e-6)=2.14, neff(lam=1.1e-6)=2.92
    gamma_0 = 2 * pi * n_eff / lam  # propagation constant

    """DC1"""
    DC1 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
    T_DC1 = DC1.transfer_matrix()

    """DC2"""
    DC2 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
    T_DC2 = DC2.transfer_matrix()

    """MZI"""
    L_MZI = 100e-6  # Length of the MZI in m
    delta_phi1 = 0.5 * pi
    delta_phi2 = 1.5 * pi

    """Rings"""
    # a_rt  # roundtrip loss
    a_rt = 1  # roundtrip loss

    # phi_rt # roundtrip phase shift
    tau_1 = 0.4  #
    tau_2 = 1  # set tau_2 to 1 for all-pass MRR

    # Radius and phirt
    R = 10e-6  # Radius of the ring in m
    phi_rt = 2 * pi * gamma_0 * R

    ring = Ring(tau_1, tau_2, a_rt, phi_rt)


    """RMZIs"""
    rmzi1 = RMZI(DC1, DC2, ring, gamma_0, L_MZI, delta_phi1)
    rmzi2 = RMZI(DC1, DC2, ring, gamma_0, L_MZI, delta_phi2)

    """T_RMZIs"""
    T_rmzi1 = rmzi1.transfer_matrix()
    T_rmzi2 = rmzi2.transfer_matrix()

    """Initializing Inputs and Outputs"""
    try:
        Input11 = np.zeros([len(lam), 2], dtype=complex)
    except:
        Input11 = np.zeros([1, 2], dtype=complex)
    Input12 = Input11.copy()
    Input21 = Input11.copy()
    Input22 = Input11.copy()

    Output11 = Input11.copy()
    Output12 = Input11.copy()
    Output21 = Input11.copy()
    Output22 = Input11.copy()

    Input11[:, 0] = 1  # MVM图像中RMZI倒转过来了
    Input11[:, 1] = 0

    """Calculating Outputs"""
    """RMZI 11"""
    Output11 = tf.matmul(T_rmzi1, Input11[..., None])
    Output11 = tf.squeeze(Output11, axis=-1)  #[100, 2, 1]变成[100, 2]
    print(f"Power Output11_1: {power(Output11[:, 0])}")
    print(f"Power Output11_2: {power(Output11[:, 1])}")

    """RMZI 12"""
    Input12[:, 0] = Input11[:, 0]
    Input12[:, 1] = Output11[:, 0]
    Output12 = tf.matmul(T_rmzi2, Input12[..., None])
    Output12 = tf.squeeze(Output12, axis=-1)
    print(f"Power Output12_1: {power(Output12[:, 0])}")
    print(f"Power Output12_2: {power(Output12[:, 1])}")

    """RMZI 22"""
    Input22[:, 0] = Output11[:, 1]
    Input22[:, 1] = 0
    Output22 = tf.matmul(T_rmzi2, Input22[..., None])
    Output22 = tf.squeeze(Output22, axis=-1)
    print(f"Power Output22_1: {power(Output22[:, 0])}")
    print(f"Power Output22_2: {power(Output22[:, 1])}")

    """RMZI 21"""
    Input21[:, 0] = Output12[:, 1]
    Input21[:, 1] = Output22[:, 0]
    Output21 = tf.matmul(T_rmzi1, Input21[..., None])
    Output21 = tf.squeeze(Output21, axis=-1)
    print(f"Power Output21_1: {power(Output21[:, 0])}")
    print(f"Power Output21_2: {power(Output21[:, 1])}")

    """Plotting Outputs"""
    # delta_phi = delta_phi / pi
    # plt.plot(lam, np.abs(Output12[:, 1]) ** 2, label=f'y1, {delta_phi:.2f}π')
    # plt.plot(lam, np.abs(Output21[:, 1]) ** 2, label=f'y2, {delta_phi:.2f}π')
    # plt.xlabel('lam / m')
    # plt.ylabel('Power (|a3|^2)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'tau={tau_1}, delta_phi={delta_phi:.2f}π.png')
    # plt.show()

    """Output Results"""
    Power_Output12 = power(Output12[:, 0])
    Power_Output21 = power(Output21[:, 0])

    # print(f"a11={}, a12={}, a21={}, a22={a22}")
    print(f"Power Output12: {Power_Output12}")
    print(f"Power Output21: {Power_Output21}")
    # run(tau_1, delta_phi)
    # plt.tight_layout()
    # plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
    # plt.close()
