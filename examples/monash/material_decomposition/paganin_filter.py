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
delta = 2.9625e-06
beta = 2.083e-07
delta_beta_ratio = delta/beta

def paganin_filter(image, pixel_size, dist, wavelength, delta_beta_ratio):
    # Convert energy to wavelength
      # wavelength in meters

    # Get image dimensions
    ny, nx = image.shape

    # Create coordinate grids
    y, x = np.ogrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    y = fftpack.fftshift(y)
    x = fftpack.fftshift(x)

    # Calculate spatial frequencies
    kx = 2 * np.pi * x / (nx * pixel_size)
    ky = 2 * np.pi * y / (ny * pixel_size)
    k = np.sqrt(kx ** 2 + ky ** 2)

    # Create Paganin filter
    denom = 1 + wavelength * dist * delta_beta_ratio * k ** 2
    paganin_filter = 1 / denom

    # Apply filter in Fourier space
    image_fft = fftpack.fft2(image)
    filtered_fft = image_fft * paganin_filter
    filtered_image = np.real(fftpack.ifft2(filtered_fft))

    return filtered_image

#filtered = paganin_filter(test_slice, psize, 0.155, wavelength, 50)
#imshow(filtered.T)

filtered2 = paganin_filter(test_slice2, psize, 0.155, wavelength, delta_beta_ratio)
imshow(filtered2.T)

