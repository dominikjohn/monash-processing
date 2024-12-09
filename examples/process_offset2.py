from monash_processing.algorithms.phase_integration import PhaseIntegrator
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.core.data_loader import DataLoader
#from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.core.volume_builder import VolumeBuilder
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask

from tqdm import tqdm
import h5py
from monash_processing.utils.utils import Utils
import pyqtgraph as pg
from pathlib import Path
import numpy as np
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2

import matplotlib


matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25 # eV
prop_distance = 0.158 #
max_angle = 364
umpa_w = 1
n_workers = 50

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

# Get number of projections (we need this for the loop)
with h5py.File(loader.h5_files[0], 'r') as f:
    num_angles = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA'].shape[0]
    print(f"Number of projections: {num_angles}")

# 2. Initialize preprocessor and UMPA processor
print("Initializing processors")
umpa_processor = UMPAProcessor(scan_path, scan_name, loader, umpa_w)

# 3. Process each projection
print("Processing projections")

# Initialize the processor
processor = UMPAProcessor(
    scan_path,
    scan_name,
    loader,
    n_workers=50
)

# Process projections
results = processor.process_projections(
    num_angles=num_angles
)

# 4. Phase integrate
print("Phase integrating")
#area_left, area_right = Utils.select_areas(loader.load_projections(projection_i=0)[0])
area_left = []
area_right = np.s_[50:-50, -80:-5]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader)
parallel_phase_integrator.integrate_parallel(num_angles, n_workers=n_workers)

################################################
import scipy

proj = 2
wavevec = 2 * np.pi * energy / (
                scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
import scipy.fft as fft
# Load dx, dy, f
dx = loader.load_processed_projection(proj, 'dx')
dy = loader.load_processed_projection(proj, 'dy')

# Create a mask for the ramp correction based on the previous user input
mask = np.zeros_like(dx, dtype=bool)
mask[area_left] = True
mask[area_right] = True

dx = BadPixelMask.correct_bad_pixels(dx)[0]
dy = BadPixelMask.correct_bad_pixels(dy)[0]

dx = np.clip(dx, -8, 8)
dy = np.clip(dy, -8, 8)

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

p_phi_corr = PhaseIntegrator.img_poly_fit(phi_corr, order=1, mask=mask)
phi_corr -= p_phi_corr

ImageViewerPhi(phi_corr, vmin=0, vmax=0.6)


################################################


volume_builder = VolumeBuilder(
    data_loader=loader,
    max_angle=max_angle,
    energy=energy,
    prop_distance=prop_distance,
    pixel_size=pixel_size,
    is_stitched=False,
    channel='phase',
    detector_tilt_deg=0,
    show_geometry=False,
    sparse_factor=20,
    is_360_deg=True
)

center_shifts = np.linspace(-1000, 2000, 15)
volume_builder.sweep_centershift(center_shifts)

center_shift = 38.8
volume_builder.reconstruct(center_shift=center_shift, chunk_count=30)