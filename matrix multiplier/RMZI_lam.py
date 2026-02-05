from main import *
from numpy import pi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

"""以波长为自变量"""
def run(tau_1, delta_phi):

    ring = Ring(tau_1, tau_2, a_rt, phi_rt)
    rmzi = RMZI(DC1, DC2, ring, gamma_0, L_MZI, delta_phi, alpha_E)
    T_rmzi = rmzi.transfer_matrix()
    a3 = tf.matmul(T_rmzi, a0[..., None])  # 相当于 a3 = T_rmzi @ Input11
    a3 = tf.squeeze(a3, axis=-1)

    # plot
    delta_phi = delta_phi / pi
    T = np.abs(a3) ** 2

    peaks1, _ = find_peaks(T[:, 0])  # 峰值索引
    valleys1, _ = find_peaks(-T[:, 0])
    print("T1 peaks:", lam[peaks1])
    print("T1 valleys:", lam[valleys1])
    peaks2, _ = find_peaks(T[:, 1])  # 峰值索引
    valleys2, _ = find_peaks(-T[:, 1])
    print("T2 peaks:", lam[peaks2])
    print("T2 valleys:", lam[valleys2])

    for i in range(len(lams_obj)):
        print(f"lam{i+1}: {lams_obj[i]}, 索引: {idx[i]}")
        print(f"T1(𝜆{i+1})={T[idx[i], 0]}")
        print(f"T2(𝜆{i+1})={T[idx[i], 1]}")

    plt.plot(lam, T[:, 0], label=f'T1, {delta_phi:.2f}π')
    plt.plot(lam, T[:, 1], label=f'T2, {delta_phi:.2f}π')
    plt.xlabel('lam / m')
    plt.ylabel('Power (|a3|^2)')
    plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'tau={tau_1}, delta_phi={delta_phi:.2f}π.png')
    # plt.show()



if __name__ == '__main__':
    alpha = 0 # power Loss coefficient in dB/cm
    alpha_E = alpha / 4.34 / 2 / 1e-2 # field Loss coefficient in 1/m

    delta_phi = 0.05 * pi  # Phase shift introduced to MZI

    start_lam = 1545e-9  # Start wavelength in meters
    end_lam = 1555e-9  # End wavelength in meters
    lam = np.arange(start_lam, end_lam, (end_lam-start_lam) / 1000)  # Wavelength in meters
    lam = lam.reshape(-1, 1)  # Reshape for broadcasting

    lams_obj = [1548e-9, 1550e-9, 1551e-9]  # Resonance wavelengths for RMZI1 and RMZI2
    idx1 = np.argmin(np.abs(lam - lams_obj[0]))
    idx2 = np.argmin(np.abs(lam - lams_obj[1]))
    idx3 = np.argmin(np.abs(lam - lams_obj[2]))
    idx = [idx1, idx2, idx3]


    Kappa_DC = np.sqrt(0.5)  # Coupling strength in um^-1
    L_DC = 11  # Length of the coupling waveguide in um
    R = 20e-6  # Radius of the ring in m
    n_eff = neff(lam)  # Effective index of the waveguide neff(lam=1.7e-6)=2.14, neff(lam=1.1e-6)=2.92
    gamma_0 = 2 * pi * n_eff / lam  # propagation constant


    """DC1"""
    DC1 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
    T_DC1 = DC1.transfer_matrix()
    a0 = np.zeros([len(lam), 2], dtype=complex)
    a0[:, 0] = 1
    a0[:, 1] = 0

    """MZI and Ring"""
    L_MZI = 100e-6  # Length of the MZI in m
    tau_1 = 0.8  #
    tau_2 = 1 # set tau_2 to 1 for all-pass MRR
    a_rt = 1  # roundtrip loss
    phase_shift = 0
    phi_rt = 2 * pi * gamma_0 * R + phase_shift  # roundtrip phase shift

    # get the m_number of the resonances
    m = (2 * pi * gamma_0 * R) / (2 * pi)
    # print("m:", m)
    # m = 189 here
    m = 189
    lam_obj = 1549.8e-9  # Objective wavelength in meters
    phase_shift_obj = 2 * pi * m - 2 * pi * (2 * pi * neff(lam_obj) / lam_obj) * R
    print(f"objective phase_shift: {phase_shift_obj / pi:.2f}π")
    #
    # ring = Ring(tau_1, tau_2, a_rt, phi_rt)
    # H_ring = ring.transfer_function_through()
    # """*ring.transfer_function_through()"""
    # a2[0] = a1[0] * np.exp(-1j * gamma_0 * L_MZI) * H_ring
    # delta_phi = 0.12*pi
    # a2[1] = a1[1] * np.exp(-1j * (gamma_0 * L_MZI + delta_phi))
    # print("Output a2:", a2[:, 0])
    # """DC2"""
    """DC2"""
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

    run(tau_1, delta_phi)
    plt.tight_layout()
    plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π, phase_shift={phase_shift / pi:.2f}π.png')
    plt.close()

    # for tau_1 in np.arange(0, a_rt, 0.1):
    #     for delta_phi in np.arange(0.4 * pi, 0.6 * pi,  0.2 * pi /10):
    #         run(tau_1, delta_phi)
    #     plt.tight_layout()
    #     plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
    #     plt.close()
    #
    # for tau_1 in np.arange(a_rt, 0.99, 0.01):
    #     for delta_phi in np.arange(0, 1 * pi,  2 * pi /10):
    #         run(tau_1, delta_phi)
    #     plt.tight_layout()
    #     plt.savefig(f'tau={tau_1}, delta_phi={delta_phi / pi:.2f}π.png')
    #     plt.close()