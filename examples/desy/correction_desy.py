from monash_processing.core.data_loader_desy import DataLoaderDesy
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

scan_base = '/asap3/petra3/gpfs/p07/2024/data/11020408/'
stitched_name = "processed/016_basel5_a_stitched_dpc/"

pixel_size = 1.28e-6  # m
energy = 40.555  # keV
prop_distance = 0.3999  #
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
ramp_correction = True
loader = DataLoaderDesy(scan_base, stitched_name)

for projection_i in tqdm(range(0, 4501)):
    T = loader.load_processed_projection(projection_i, 'T_stitched', format='tif', simple_format=True)

    # gauss_size = 61
    # I = cv2.GaussianBlur(I, (gauss_size, gauss_size), 0)

    dx = loader.load_processed_projection(projection_i, 'dx_stitched', simple_format=True)
    dy = loader.load_processed_projection(projection_i, 'dy_stitched', simple_format=True)

    dx = BadPixelMask.correct_bad_pixels(dx)[0]
    dy = BadPixelMask.correct_bad_pixels(dy)[0]

    mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
    mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

    k = fft.fftfreq(mdx.shape[1])
    l = fft.fftfreq(mdy.shape[0])
    k[k == 0] = 1e-10
    l[l == 0] = 1e-10
    k, l = np.meshgrid(k, l)

    delta_mu = 3600

    conversion_factor = (wavevec / prop_distance) * (pixel_size ** 2)
    ft = fft.fft2(mdx + 1j * mdy, workers=2)
    phi_raw = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)), workers=2)
    phi_umpa = np.real(phi_raw)[dx.shape[0]:, dy.shape[1]:] * conversion_factor
    phi_coarse = -np.log(T) * wavevec * delta_mu * pixel_size ** 2

    mask = np.zeros_like(phi_umpa, dtype=bool)
    mask[:, 0:2000] = True
    mask[:, -2000:-5] = True
    phi_umpa_cor = phi_umpa - PhaseIntegrator.img_poly_fit(phi_umpa, order=1, mask=mask)

    phi_coarse_cor = phi_coarse - np.average(phi_coarse[mask])

    m_umpa = PhaseIntegrator.sym_mirror_im(phi_umpa_cor, 'reflect')
    m_coarse = PhaseIntegrator.sym_mirror_im(phi_coarse_cor, 'reflect')

    phi_umpa_ft = fft.fft2(m_umpa)
    phi_coarse_ft = fft.fft2(m_coarse)

    # very low value -> UMPA (< 0.01)
    # high value (> 0.01): mostly coarse
    #sigma = .013
    sigma = 0.035
    weighted_phi_ft = phi_umpa_ft * (1 - np.exp(-np.sqrt(k ** 2 + l ** 2) / (2 * sigma ** 2))) + phi_coarse_ft * np.exp(
        -np.sqrt(k ** 2 + l ** 2) / (2 * sigma ** 2))
    final_phi = np.real(fft.ifft2(weighted_phi_ft))[phi_umpa_cor.shape[0]:, phi_umpa_cor.shape[1]:]
    #imshow(final_phi)

    loader.save_tiff('corrected_phi', projection_i, final_phi)
    loader.save_tiff('original_phi', projection_i, phi_umpa_cor)