from monash_processing.core.data_loader import DataLoader
from monash_processing.algorithms.phase_integration import PhaseIntegrator
from pathlib import Path
import numpy as np
from scipy import fft
import scipy.constants
import matplotlib
from monash_processing.utils.ImageViewer import ImageViewer as imshow

import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
print(matplotlib.get_backend())  # Verify it's using TkAgg
import matplotlib.pyplot as plt

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25 # keV
prop_distance = 0.158 #
max_angle = 182
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
ramp_correction = False

print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

projection_i = 0

raw_I = np.average(loader.load_projections(projection_i=projection_i), axis=0)
imshow(raw_I)


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

p_phi_corr = _img_poly_fit(phi_corr, order=1, mask=mask)
phi_corr -= p_phi_corr

if np.percentile(phi_corr, 99) > 1E3:
    print(f'Integration of projection {projection_i} failed!')
    phi_corr = np.zeros_like(phi_corr)

loader.save_tiff('phi', projection_i, phi_corr)