import numpy as np
from main import *
pi = np.pi

alpha = 0 # power Loss coefficient in dB/cm
alpha_E = alpha / 4.34 / 2 / 1e-2 # field Loss coefficient in 1/m

delta_phi = 0.05 * pi  # Phase shift introduced to MZI

# MZI and Ring parameters
L_MZI = 100e-6  # Length of the MZI in m
tau_1 = 0.8  #
tau_2 = 1 # set tau_2 to 1 for all-pass MRR
a_rt = 1  # roundtrip loss

start_lam = 1547.5e-9  # Start wavelength in meters
end_lam = 1548.5e-9  # End wavelength in meters
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

"DC1"
DC1 = DC(Kappa=Kappa_DC, L=L_DC, gamma_0=gamma_0)
"DC2"
DC2 = DC1

"""define RMZI 1, Phase_shift = -1.43 pi"""
phase_shift = -1.43 * pi

"Ring and RMZI"
phi_rt = 2 * pi * gamma_0 * R + phase_shift  # roundtrip phase shift
ring1 = Ring(tau_1, tau_2, a_rt, phi_rt)
rmzi1 = RMZI(DC1, DC2, ring1, gamma_0, L_MZI, delta_phi, alpha_E)

"""define RMZI 2, Phase_shift = -0.70π"""
phase_shift = -0.7 * pi

"Ring and RMZI"
phi_rt = 2 * pi * gamma_0 * R + phase_shift  # roundtrip phase shift
ring2 = Ring(tau_1, tau_2, a_rt, phi_rt)
rmzi2 = RMZI(DC1, DC2, ring2, gamma_0, L_MZI, delta_phi, alpha_E)

"""define RMZI 3, Phase_shift = -0.70π"""
phase_shift = 0

"Ring and RMZI"
phi_rt = 2 * pi * gamma_0 * R + phase_shift  # roundtrip phase shift
ring3 = Ring(tau_1, tau_2, a_rt, phi_rt)
rmzi3 = RMZI(DC1, DC2, ring3, gamma_0, L_MZI, delta_phi, alpha_E)

RMZI_list = [[rmzi1, rmzi2, rmzi3],
                [rmzi2, rmzi1, rmzi3],
                [rmzi3, rmzi2, rmzi1]]

mvm = MVM(3, RMZI_list)

# Output.shape = [3, 3, 1000, 2]
Output = mvm.calculate_output(1)

import matplotlib.pyplot as plt

# 获取三条曲线的数据
curve1 = Output[0, 2, :, 0]  # 第一条曲线
curve2 = Output[1, 2, :, 0]  # 第二条曲线
curve3 = Output[2, 2, :, 0]  # 第三条曲线

# 创建波长轴（对应1000个点）
wavelength = lam.flatten()  # 使用之前定义的波长数组

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(wavelength * 1e9, np.abs(curve1) ** 2, label='Output1', linewidth=2)
plt.plot(wavelength * 1e9, np.abs(curve2) ** 2, label='Output2', linewidth=2)
plt.plot(wavelength * 1e9, np.abs(curve3) ** 2, label='Output3', linewidth=2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.title('MVM Output Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MVM_Output_Curves.png', dpi=300, bbox_inches='tight')
plt.show()


