from monash_processing.core.data_loader import DataLoader
from monash_processing.algorithms.phase_integration import PhaseIntegrator
from pathlib import Path
import numpy as np
from scipy import fft
import scipy.constants
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2
import matplotlib
matplotlib.use('TkAgg', force=True)
from tqdm import tqdm

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25 # keV
prop_distance = 0.158 #
max_angle = 182
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
ramp_correction = False
antisym_mirror = False
blurring = False

print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

projection_i = 0

ramp_correction = True
for projection_i in tqdm(range(0, 20)):

    raw_I = np.average(loader.load_projections(projection_i=projection_i), axis=0)
    I = loader.perform_flatfield_correction(raw_I, np.average(flat_fields, axis=0), dark_current)

    if blurring:
        gauss_size = 61
        blurred_I = cv2.GaussianBlur(I, (gauss_size, gauss_size), 0)
    else:
        blurred_I = I

    background = np.s_[20:-20, -100:-10]
    cutter = np.s_[5:-5, 5:-5]
    blurred_I = blurred_I[cutter]
    dx = loader.load_processed_projection(projection_i, 'dx')
    dy = loader.load_processed_projection(projection_i, 'dy')
    f = loader.load_processed_projection(projection_i, 'f')

    dx = BadPixelMask.correct_bad_pixels(dx)[0]
    dy = BadPixelMask.correct_bad_pixels(dy)[0]

    mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
    mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

    k = fft.fftfreq(mdx.shape[1])
    l = fft.fftfreq(mdy.shape[0])
    k[k == 0] = 1e-10
    l[l == 0] = 1e-10
    k, l = np.meshgrid(k, l)

    delta_mu = 2000

    conversion_factor = (wavevec / prop_distance) * (pixel_size ** 2)
    ft = fft.fft2(mdx + 1j * mdy, workers=2)
    phi_raw = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)), workers=2)
    phi_umpa = np.real(phi_raw)[dx.shape[0]:, dy.shape[1]:] * conversion_factor
    phi_coarse = -np.log(blurred_I) * wavevec * delta_mu * pixel_size**2

    phi_umpa_cor = phi_umpa - np.average(phi_umpa[background])
    phi_coarse_cor = phi_coarse - np.average(phi_coarse[background])

    def cor_bg(image, background):
        return image - np.average(image[background])

    m_umpa = PhaseIntegrator.sym_mirror_im(phi_umpa_cor, 'reflect')
    m_coarse = PhaseIntegrator.sym_mirror_im(phi_coarse_cor, 'reflect')

    phi_umpa_ft = fft.fft2(m_umpa)
    phi_coarse_ft = fft.fft2(m_coarse)

    # very low value -> UMPA (< 0.01)
    # high value (> 0.01): mostly coarse
    sigma = .02
    weighted_phi_ft = phi_umpa_ft * (1-np.exp(-np.sqrt(k**2 + l**2)/(2*sigma**2))) + phi_coarse_ft * np.exp(-np.sqrt(k**2 + l**2)/(2*sigma**2))
    final_phi = np.real(fft.ifft2(weighted_phi_ft))[phi_umpa_cor.shape[0]:, phi_umpa_cor.shape[1]:]

    loader.save_tiff('corrected_phi', projection_i, final_phi)
    loader.save_tiff('original_phi', projection_i, phi_umpa_cor)

    if ramp_correction:
        # Create a mask for the ramp correction based on the previous user input
        mask = np.zeros_like(final_phi, dtype=bool)
        mask[background] = True
        final_phi = final_phi - PhaseIntegrator.img_poly_fit(final_phi, order=1, mask=mask)
        phi_umpa_cor = phi_umpa_cor - PhaseIntegrator.img_poly_fit(phi_umpa_cor, order=1, mask=mask)

        loader.save_tiff('corrected_phi_ramp_cor', projection_i, final_phi)
        loader.save_tiff('original_phi_ramp_cor', projection_i, phi_umpa_cor)
