import numpy as np

from scipy.special import voigt_profile


def simulate(x, x_stick, I_stick, sigma, gamma=0, N_bin=50):
    x_stick = np.array(x_stick)[:, np.newaxis]
    I_stick = np.array(I_stick)[:, np.newaxis]
    Ii = I_stick * np.exp(-((x - x_stick) ** 2) / (2 * sigma**2))
    # I_x = np.sum(Ii, axis=0)

    dx = np.median(np.diff(x))
    x2 = x + dx / N_bin * (np.arange(N_bin)[:, np.newaxis] - (N_bin - 1) / 2)
    x2 = x2.T.ravel()
    # Ii2 = I_stick * np.exp(-(x2 - x_stick)**2/(2*sigma**2))
    Ii2 = I_stick * voigt_profile(x2 - x_stick, sigma, gamma)
    I_x2 = np.sum(Ii2, axis=0)

    x3 = x2.reshape(-1, N_bin).mean(axis=1)
    I_x3 = I_x2.reshape(-1, N_bin).mean(axis=1)

    # assert np.all(x == x3)

    return I_x3
