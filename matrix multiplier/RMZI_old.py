import numpy as np
from main import *
from numpy import pi
import matplotlib.pyplot as plt
import sympy as sp

"""以phi_rt为自变量"""
def run(tau_1, delta_phi):
    a1 = T_DC1 @ a0
    print("Output a1:", a1)

    ring = Ring(tau_1, tau_2, a_rt, phi_rt)
    H_ring = ring.transfer_function_through()
    a2[0] = a1[0] * np.exp(1j * gamma_0 * L_MZI) * H_ring
    a2[1] = a1[1] * np.exp(1j * (gamma_0 * L_MZI - delta_phi))
    print("Output a2:", a2[:, 0])

    a3 = T_DC2 @ a2
    print("Output a3:", a3[:, 0])

    # plot
    delta_phi = delta_phi / pi
    plt.plot(phi_rt, np.abs(a3[0]) ** 2, label=f'T1, {delta_phi:.2f}π')
    plt.plot(phi_rt, np.abs(a3[1]) ** 2, label=f'T2, {delta_phi:.2f}π')
    plt.xlabel('phi_rt / rad')
    plt.ylabel('Power (|a3|^2)')
    plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'tau={tau_1}, delta_phi={delta_phi:.2f}π.png')
    # plt.show()

if __name__ == '__main__':
    lam = 1550e-9  # Wavelength in meters
    Kappa_DC = np.sqrt(0.4)  # Coupling strength in um^-1
    L_DC = 11  # Length of the coupling waveguide in um

    n_eff = 2.4  # Effective index of the waveguide
    gamma_0 = 2 * pi * n_eff / lam  # propagation constant
    """DC1"""
    DC1 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
    T_DC1 = DC1.transfer_matrix()
    print("Transfer Matrix:", T_DC1)
    a0 = np.zeros([2, 1])
    a1 = a0
    a0[0] = 1
    a0[1] = 0
    a1 = T_DC1 @ a0

    """MZI and Ring"""
    L_MZI = 100e-6  # Length of the MZI in m
    tau_1 = 0.4  #
    tau_2 = 1 # set tau_2 to 1 for all-pass MRR
    a_rt = 1.  # roundtrip loss
    phi_rt = np.arange(-pi, pi, 2 * pi / 100)
    a2 = np.zeros([2, len(phi_rt)], dtype=complex)
    #
    # ring = Ring(tau_1, tau_2, a_rt, phi_rt)
    # H_ring = ring.transfer_function_through()
    # """*ring.transfer_function_through()"""
    # a2[0] = a1[0] * np.exp(-1j * gamma_0 * L_MZI) * H_ring
    # delta_phi = 0.12*pi
    # a2[1] = a1[1] * np.exp(-1j * (gamma_0 * L_MZI + delta_phi))
    # print("Output a2:", a2[:, 0])
    # """DC2"""
    DC2 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
    T_DC2 = DC2.transfer_matrix()
    # a3 = T_DC2 @ a2
    # print("Output a3:", a3[:, 0])
    # plt.plot(phi_rt, np.abs(a3[0]) ** 2, label='Output 1 (a3[0])')
    # plt.plot(phi_rt, np.abs(a3[1]) ** 2, label='Output 2 (a3[1])')
    # plt.xlabel('phi_rt / rad')
    # plt.ylabel('Power (|a3|^2)')
    # plt.legend()
    # plt.savefig('Power Transmission.png')
    # plt.show()
    tau_1 = 0.4
    delta_phi = 0.4 * pi
    run(tau_1, delta_phi)
    plt.tight_layout()
    plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
    plt.close()
    for tau_1 in np.arange(0, a_rt, 0.1):
        for delta_phi in np.arange(0.4 * pi, 0.6 * pi,  2 * pi /100):
            run(tau_1, delta_phi)
        plt.tight_layout()
        plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
        plt.close()

    for tau_1 in np.arange(a_rt, 0.99, 0.01):
        for delta_phi in np.arange(0, 1 * pi,  2 * pi /10):
            run(tau_1, delta_phi)
        plt.tight_layout()
        plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
        plt.close()