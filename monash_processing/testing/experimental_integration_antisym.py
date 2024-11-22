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
energy = 25 # eV
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

gauss_size = 301
blurred_I = cv2.GaussianBlur(I, (gauss_size, gauss_size), 0)
#imshow(blurred_I)

blurred_I = blurred_I / np.max(blurred_I)

cutter = np.s_[5:-5, 5:-5]
blurred_I = blurred_I[cutter]
dx = loader.load_processed_projection(projection_i, 'dx')
dy = loader.load_processed_projection(projection_i, 'dy')
f = loader.load_processed_projection(projection_i, 'f')

# Set a threshold value for the cleanup based on the UMPA error map (99th percentile)
thl = np.round(np.percentile(f, 99))
dx = np.clip(PhaseIntegrator.cleanup_rio(dx, f, thl), -8, 8)
dy = np.clip(PhaseIntegrator.cleanup_rio(dy, f, thl), -8, 8)

mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

mirror_I = np.pad(blurred_I, ((blurred_I.shape[0], 0), (blurred_I.shape[1], 0)), mode='reflect')

k = fft.fftfreq(mdx.shape[1])
l = fft.fftfreq(mdy.shape[0])
k[k == 0] = 1e-10
l[l == 0] = 1e-10
k, l = np.meshgrid(k, l)

delta_mu = 500
epsilon_values = [2E-4, 3E-4, 5E-4]
processed_images = []
titles = []

for epsilon in epsilon_values:
    term = mdx + 1j * mdy
    term_cor = term + epsilon * wavevec * delta_mu * np.log(mirror_I)

    denominator = (2 * np.pi * 1j) * (k + 1j * l)
    denominator_cor = denominator + 2 * np.pi * epsilon

    ft = fft.fft2(term, workers=2)
    ft_cor = fft.fft2(term, workers=2)

    phi_final = np.real(fft.ifft2(ft / denominator, workers=2)) * (wavevec / prop_distance) * (pixel_size ** 2)

    phi_final_cor = np.real(fft.ifft2(ft_cor / denominator_cor, workers=2)) * (wavevec / prop_distance) * (pixel_size ** 2)

    processed_images.append(phi_final_cor[:] - np.average(phi_final_cor))
    titles.append(f'epsilon = {epsilon:.2e}')

processed_images.insert(0, phi_final - np.average(phi_final))
titles.insert(0, f'no correction')

# Create and show the viewer
viewer = SideBySideViewer(processed_images, titles)
viewer.show()

def cor_bg(image, background):
    return image - np.average(image[background])

constant = (wavevec / prop_distance) * (pixel_size ** 2)

Phi_1 = np.real(fft.ifft2(fft.fft2(mdx)/(1j * k)))[dx.shape[0]:, :dy.shape[1]] * (wavevec / prop_distance) * (pixel_size ** 2)
Phi_2 = np.real(fft.ifft2(fft.fft2(mdy)/(1j * l)))[dx.shape[0]:, :dy.shape[1]] * (wavevec / prop_distance) * (pixel_size ** 2)
Phi_1_avg = np.average(Phi_1)

background = np.s_[10:-10, 10:100]
imshow(cor_bg(Phi_1, background))
imshow(cor_bg(Phi_2, background))

phi_coarse = np.log(mirror_I) * epsilon * wavevec * delta_mu
epsilon_sq = 1E-6

numerator = 1j * k * fft.fft2(mdx) + 1j * l * fft.fft2(mdy) - epsilon_sq * fft.fft2(phi_coarse)
denominator = epsilon_sq + k**2 + l**2

Phi_filtered = -np.real(fft.ifft2(numerator / denominator))[dx.shape[0]:, :dy.shape[1]] * constant
#Phi_filtered = Phi_filtered - np.mean(Phi_filtered[background])
ImageViewerPhi(Phi_filtered)


epsilon_values = [0, 1E-6, 5E-6, 1E-5]
image_list = []
for epsilon in epsilon_values:
    phi_coarse = np.log(mirror_I) * epsilon * wavevec * delta_mu
    epsilon_sq = epsilon**2

    numerator = 1j * k * fft.fft2(mdx) + 1j * l * fft.fft2(mdy) - epsilon_sq * fft.fft2(phi_coarse)
    denominator = epsilon_sq + k**2 + l**2

    Phi_filtered = -np.real(fft.ifft2(numerator / denominator))[dx.shape[0]:, :dy.shape[1]] * constant
    Phi_filtered = Phi_filtered - np.mean(Phi_filtered[background])

    image_list.append(Phi_filtered)

titles = [f'epsilon = {epsilon:.2e}' for epsilon in epsilon_values]
viewer = SideBySideViewer(image_list, titles)


########################################################################

phi_coarse = np.log(mirror_I) * epsilon * wavevec * delta_mu
epsilon_sq = epsilon**2

numerator = 1j * k * fft.fft2(mdx) + 1j * l * fft.fft2(mdy) - epsilon_sq * fft.fft2(phi_coarse)
denominator = epsilon_sq + k**2 + l**2

Phi_filtered = -np.real(fft.ifft2(numerator / denominator))[dx.shape[0]:, :dy.shape[1]] * constant
Phi_filtered = Phi_filtered - np.mean(Phi_filtered[background])

image_list.append(Phi_filtered)






########################################################################

S_1 = mdx + epsilon * phi_coarse
Phi_1_filter = np.real(fft.ifft2(fft.fft2(mdx)/(1j * k + epsilon)))[dx.shape[0]:, :dy.shape[1]]
Phi_1_filter_avg = np.average(Phi_1_filter)

viewer = SideBySideViewer([, Phi_1_filter-Phi_1_filter_avg], ['Phi_1', 'Phi_1_filter'])
viewer.show()

Phi_2 = np.real(fft.ifft2(fft.fft2(mdy)/(1j * l)))[dx.shape[0]:, :dy.shape[1]]
average = (Phi_1 + Phi_2) / 2