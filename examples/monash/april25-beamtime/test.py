from scipy.constants import h, c, k
import numpy as np



def planck(lam, T):
    """ Returns the spectral radiance of a black body at temperature T.

    Returns the spectral radiance, B(lam, T), in W.sr-1.m-2 of a black body
    at temperature T (in K) at a wavelength lam (in nm), using Planck's law.

    """
    lam_m = lam / 1.e9
    fac = h * c / lam_m / k / T
    B = 2 * h * c ** 2 / lam_m ** 5 / (np.exp(fac) - 1)
    return B

wavelengths = np.arange(380, 780, 1)
spectrum = planck(wavelengths, 6500)
total = np.sum(spectrum)
import matplotlib.pyplot as plt
plt.plot(wavelengths, spectrum/total)
plt.show()