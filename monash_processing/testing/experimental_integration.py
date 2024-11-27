from monash_processing.core.data_loader import DataLoader
from monash_processing.algorithms.phase_integration import PhaseIntegrator
from pathlib import Path
import numpy as np
from scipy import fft
import scipy.constants
import matplotlib
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2

import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
print(matplotlib.get_backend())  # Verify it's using TkAgg
import matplotlib.pyplot as plt

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25e3 # eV
prop_distance = 0.158 #
max_angle = 182
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
ramp_correction = False
antisym_mirror = False

print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

projection_i = 0

raw_I = np.average(loader.load_projections(projection_i=projection_i), axis=0)
I = loader.perform_flatfield_correction(raw_I, np.average(flat_fields, axis=0), dark_current)

imshow(I)

gauss_size = 301
blurred_I = cv2.GaussianBlur(I, (gauss_size, gauss_size), 0)
blurred_I = blurred_I / np.max(blurred_I)
#imshow(blurred_I)

cutter = np.s_[5:-5, 5:-5]
blurred_I = blurred_I[cutter]
dx = loader.load_processed_projection(projection_i, 'dx')
dy = loader.load_processed_projection(projection_i, 'dy')
f = loader.load_processed_projection(projection_i, 'f')

# Set a threshold value for the cleanup based on the UMPA error map (99th percentile)
thl = np.round(np.percentile(f, 99))
dx = np.clip(PhaseIntegrator.cleanup_rio(dx, f, thl), -8, 8)
dy = np.clip(PhaseIntegrator.cleanup_rio(dy, f, thl), -8, 8)

if ramp_correction:
    area_left = np.s_[100:-100, 20:120]
    area_right = np.s_[100:-100, -120:-20]
    # Create a mask for the ramp correction based on the previous user input
    mask = np.zeros_like(dx, dtype=bool)
    mask[area_left] = True
    mask[area_right] = True
    dx -= PhaseIntegrator.img_poly_fit(dx, order=1, mask=mask)
    dy -= PhaseIntegrator.img_poly_fit(dy, order=1, mask=mask)

k = fft.fftfreq(dx.shape[1])
l = fft.fftfreq(dx.shape[0])
k[k == 0] = 1E-10
l[l == 0] = 1E-10
k, l = np.meshgrid(k, l)

epsilon_values = [2E-4, 3E-4, 5E-4]
processed_images = []
titles = []

for eps in epsilon_values:
    # Your existing processing code here
    epsilon = eps
    term = dx + 1j * dy
    term_cor = term + epsilon * wavevec * delta_mu * np.log(blurred_I)

    denominator = (2 * np.pi * 1j) * (k + 1j * l)
    denominator_cor = denominator + 2 * np.pi * epsilon

    ft = fft.fft2(term, workers=2)
    ft_cor = fft.fft2(term, workers=2)

    phi_raw = fft.ifft2(ft / denominator, workers=2)
    phi_final = np.real(phi_raw) * (wavevec / prop_distance) * (pixel_size ** 2)

    phi_raw_cor = fft.ifft2(ft_cor / denominator_cor, workers=2)
    phi_final_cor = np.real(phi_raw_cor) * (wavevec / prop_distance) * (pixel_size ** 2)

    processed_images.append(phi_final_cor - np.average(phi_final_cor))
    titles.append(f'epsilon = {eps:.2e}')

mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

k = fft.fftfreq(mdx.shape[1])
l = fft.fftfreq(mdy.shape[0])
k[k == 0] = 1e-10
l[l == 0] = 1e-10
k, l = np.meshgrid(k, l)

ft = fft.fft2(mdx + 1j * mdy)
phi_raw_mirror = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)))
phi_final_mirror = np.real(phi_raw_mirror)[dx.shape[0]:, dy.shape[1]:] * (wavevec / prop_distance) * (pixel_size ** 2)

processed_images.insert(0, phi_final_cor - np.average(phi_final_cor))
titles.insert(0, f'no correction')

processed_images.insert(1, phi_final_mirror - np.average(phi_final_mirror))
titles.insert(1, f'antisym no correction')

# Create and show the viewer
viewer = SideBySideViewer(processed_images, titles)
viewer.show()

loader.save_tiff('phi', projection_i, phi_corr)

if antisym_mirror:
    mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
    mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

    k = fft.fftfreq(mdx.shape[1])
    l = fft.fftfreq(mdy.shape[0])
    k[k == 0] = 1e-10
    l[l == 0] = 1e-10
    k, l = np.meshgrid(k, l)

    ft = fft.fft2(mdx + 1j * mdy, workers=2)
    phi_raw = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)), workers=2)
    phi_raw = np.real(phi_raw)[dx.shape[0]:, dy.shape[1]:]

    phi_corr = phi_raw * (wavevec / prop_distance) * (pixel_size ** 2)

    #p_phi_corr = PhaseIntegrator.img_poly_fit(phi_corr, order=1, mask=mask)
    #phi_corr -= p_phi_corr