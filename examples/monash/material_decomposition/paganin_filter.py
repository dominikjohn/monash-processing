import numpy as np
from scipy import fftpack
from scipy.special import betaln

from monash_processing.core.data_loader import DataLoader
import scipy.constants
import numpy as np
from scipy import fftpack

from pathlib import Path
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2
from skimage.measure import block_reduce
import matplotlib
import cupy as cp

import os
from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis

matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt

binning_factor = 1
psize = 1.444e-6 * binning_factor

energy = 25000
energy_keV = energy / 1000
wavelength = 1.24e-9 / energy_keV

#loader = DataLoader(Path("/data/mct/22203/"), "P6_Manual")
#loader = DataLoader(Path("/data/mct/22203/"), "K3_3H_Manual")
loader = DataLoader(Path("/data/mct/22203/"), "K3_1N")
#edensity_volume = loader.load_reconstruction('recon_phase', binning_factor=1)
mu_volume = loader.load_reconstruction('recon_att', binning_factor=1)

# imshow(mu_volume.transpose(1,2,0)[1200].T)

test_slice = mu_volume.transpose(1, 2, 0)[1200]
test_slice2 = mu_volume[1200]
imshow(test_slice.T)

# Lead
#delta = 2.9625e-06
#beta = 2.083e-07
#delta_beta_ratio = delta/beta

delta_beta_ratio = 12.6896


def paganin_filter(image, pixel_size, dist, wavelength, delta_beta_ratio):
    # Get image dimensions
    ny, nx = image.shape

    # Calculate frequencies using fftfreq
    delta_x = pixel_size / (2 * np.pi)
    kx = np.fft.fftfreq(nx, d=delta_x)
    ky = np.fft.fftfreq(ny, d=delta_x)

    # Create 2D frequency grid
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_squared = kx_grid ** 2 + ky_grid ** 2

    # Create Paganin filter with corrected formula
    # Since we're using delta_beta_ratio = delta/beta, need to multiply by 1/(4π)
    denom = 1 + dist * wavelength * (delta_beta_ratio / (4 * np.pi)) * k_squared
    paganin_filter = 1 / denom

    # Apply filter in Fourier space
    image_fft = np.fft.fft2(image)
    filtered_fft = image_fft * paganin_filter
    filtered_image = np.real(np.fft.ifft2(filtered_fft))

    return filtered_image


def paganin_filter_gpu(image, pixel_size, dist, wavelength, delta_beta_ratio):
    """
    GPU-accelerated implementation of the Paganin phase retrieval filter.

    Parameters:
    -----------
    image : numpy.ndarray
        Input intensity image
    pi  xel_size : float
        Detector pixel size in meters
    dist : float
        Sample-to-detector distance in meters
    wavelength : float
        X-ray wavelength in meters
    delta_beta_ratio : float
        Ratio of refractive index decrement to absorption index

    Returns:
    --------
    numpy.ndarray
        Retrieved phase image
    """
    # Transfer image to GPU
    image_gpu = cp.asarray(image)

    # Get image dimensions
    ny, nx = image_gpu.shape

    # Create coordinate grids on GPU
    y, x = cp.ogrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    y = cp.fft.fftshift(y)
    x = cp.fft.fftshift(x)

    # Calculate spatial frequencies
    kx = 2 * cp.pi * x / (nx * pixel_size)
    ky = 2 * cp.pi * y / (ny * pixel_size)
    k = cp.sqrt(kx ** 2 + ky ** 2)

    # Create Paganin filter
    denom = 1 + wavelength * dist * delta_beta_ratio * k ** 2
    paganin_filter = 1 / denom

    # Apply filter in Fourier space
    image_fft = cp.fft.fft2(image_gpu)
    filtered_fft = image_fft * paganin_filter
    filtered_image_gpu = cp.real(cp.fft.ifft2(filtered_fft))

    # Transfer result back to CPU
    filtered_image = cp.asnumpy(filtered_image_gpu)

    # Clean up GPU memory
    del image_gpu, y, x, kx, ky, k, denom, paganin_filter, image_fft, filtered_fft, filtered_image_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return filtered_image

#filtered = paganin_filter(test_slice, psize, 0.155, wavelength, 50)
#imshow(filtered.T)

filtered2 = paganin_filter(test_slice2, psize, 0.155, wavelength, delta_beta_ratio)
imshow(filtered2.T)

