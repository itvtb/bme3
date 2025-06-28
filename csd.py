import numpy as np
import pandas as pd
from scipy.linalg import lstsq, eigvals

def read_data_csv(filename):
    df = pd.read_csv(filename)
    freq = df.iloc[:, 0].values  # cột 0: frequency (Hz)
    mag = df.iloc[:, 1].values   # cột 1: magnitude (trở kháng)
    phase_deg = df.iloc[:, 2].values  # cột 2: phase (độ)
    phase_rad = np.deg2rad(phase_deg)  # đổi sang radian
    Z = mag * np.exp(1j * phase_rad)  # giá trị phức
    return freq, Z

def vector_fit(s, H, n_poles=6, n_iter=10):
    poles = -np.logspace(np.log10(abs(s[0])), np.log10(abs(s[-1])), n_poles) * (1 + 1j)
    for iteration in range(n_iter):
        m = len(s)
        A = np.zeros((m, 2*n_poles + 2), dtype=complex)
        for k in range(n_poles):
            A[:, k] = 1/(s - poles[k])
            A[:, n_poles + k] = -H / (s - poles[k])
        A[:, -2] = 1
        A[:, -1] = -H
        x, _, _, _ = lstsq(A, H)
        residues = x[:n_poles]
        residues_correction = x[n_poles:2*n_poles]
        d = x[-2]
        h = x[-1]
        poles_matrix = np.diag(poles) - np.outer(residues_correction, np.ones(n_poles))
        poles = eigvals(poles_matrix)
        print(f"Iter {iteration+1}/{n_iter} poles (real part): {poles.real}")
    return residues, poles, d, h

def main():
    filename = '333.csv'
    freq, Z = read_data_csv(filename)
    s = 1j * 2 * np.pi * freq
    residues, poles, d, h = vector_fit(s, Z, n_poles=6, n_iter=10)
    print("Kết quả Vector Fitting:")
    print("Residues:", residues)
    print("Poles:", poles)
    print("d:", d)
    print("h:", h)

if __name__ == '__main__':
    main()
