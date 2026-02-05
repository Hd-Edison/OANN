import numpy as np
pi = np.pi
def neff(lam):
    # calculate effective index based on wavelength
    # 500nm * 220nm Si waveguide, substrate = cover = air
    return -1.3e6*lam + 4.35  # Example linear dependence

if __name__ == '__main__':
    print(0.9997+0.9995 * 0.25)
    print(0.9997 * 0.25 + 0.9995)
    print(0.9786357+0.24996534)