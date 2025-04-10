from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.core.data_loader import DataLoader
from monash_processing.core.volume_builder import VolumeBuilder
import h5py
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_3H_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25000 # eV
prop_distance = 0.158 #
max_angle = 365
umpa_w = 1
n_workers = 100

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)
angles = np.mean(loader.load_angles(), axis=0)
angle_step = np.diff(angles).mean()
print('Angle step:', angle_step)
index_0 = np.argmin(np.abs(angles - 0))
index_180 = np.argmin(np.abs(angles - 180))
print('Index at 0°:', index_0)
print('Index at 180°:', index_180)

# Get number of projections (we need this for the loop)
with h5py.File(loader.h5_files[0], 'r') as f:
    num_angles = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA'].shape[0]
    print(f"Number of projections: {num_angles}")

# 3. Process each projection
print("Processing projections")

# Initialize the processor
processor = UMPAProcessor(
    scan_path,
    scan_name,
    loader,
    umpa_w,
    n_workers=50,
    #slicing=np.s_[..., :, :]
    slicing=np.s_[..., :, :]
    #slicing=np.s_[..., 800:-500, :]
)

# Process projections
results = processor.process_projections(
    num_angles=num_angles
)

area_left = np.s_[:, 5:80]
area_right = np.s_[:, -80:-5]

max_index = int(np.round(180 / angle_step))
print('Uppermost projection index: ', max_index)

center_shift_list = np.linspace(805, 810, 5)
for center_shift in center_shift_list:
    suffix = f'{(2 * center_shift):.2f}'
    stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=center_shift / 2, slices=(300, 310), suffix=suffix, window_size=umpa_w)
    stitcher.process_and_save_range(index_0, index_180, 'dx')
    stitcher.process_and_save_range(index_0, index_180, 'dy')
    #stitcher.process_and_save_range(index_0, index_180, 'T_raw')
    parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                        loader, stitched=True, suffix=suffix, window_size=umpa_w)
    parallel_phase_integrator.integrate_parallel(max_index+1, n_workers=n_workers)
    volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='phase',
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        suffix=suffix,
        window_size=umpa_w,
    )
    volume_builder.reconstruct(center_shift=0, chunk_count=1, custom_folder='offset_sweep', slice_range=(2,4))

best_value = 805
blending = True # whether to blend linearly
stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=best_value / 2, format='tif', window_size=umpa_w)
stitcher.process_and_save_range(index_0, index_180, 'dx', blending=True)
stitcher.process_and_save_range(index_0, index_180, 'dy', blending=True)
stitcher.process_and_save_range(index_0, index_180, 'T')
stitcher.process_and_save_range(index_0, index_180, 'df')

#stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=best_value / 2, format='tif', window_size=umpa_w)
#stitcher.process_and_save_range(index_0, index_180, 'df_positive')
area_left = np.s_[:, 5:80]
area_right = np.s_[:, -80:-5]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, window_size=umpa_w, stitched=True)
parallel_phase_integrator.integrate_parallel(max_index+1, n_workers=n_workers)

import scipy

mu_st = 0.567 * 100  # measurement
#delta_st = 0.377e-6  # measurement
delta_st = 0.3305e-6
mu_lead = 550.28 * 100  # 1 / m, source NIST
delta_lead = 2.9625e-06  # confirmed via NIST

wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
beta_st = mu_st / (2 * wavevec)
beta_lead = mu_lead / (2 * wavevec)

print((delta_st-delta_lead)/(beta_st-beta_lead))

for i in range(1796):
    T_stitched = loader.load_processed_projection(i, 'T_stitched', subfolder=f'umpa_window{umpa_w}', format='tif')
    wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)

    ny, nx = T_stitched.shape

    # Calculate frequencies using fftfreq
    delta_x = pixel_size / (2 * np.pi)
    kx = np.fft.fftfreq(nx, d=delta_x)
    ky = np.fft.fftfreq(ny, d=delta_x)

    # Create 2D frequency grid
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_squared = kx_grid ** 2 + ky_grid ** 2

    image_fft = np.fft.fft2(T_stitched)

    denom = (prop_distance * (delta_st - delta_lead) / (mu_st - mu_lead)) * k_squared + 1
    filter = 1 / denom

    filtered_fft = image_fft * filter
    absorption = np.real(np.fft.ifft2(filtered_fft))
    #log_image = np.log(np.real(np.fft.ifft2(filtered_fft)))
    #thickness = -log_image / (mu_2 - mu_enc)
    #thickness = -log_image / (mu_st - mu_lead)

    loader.save_tiff('absorption', i, absorption, subfolder=f'umpa_window{umpa_w}')

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='phase',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        window_size=umpa_w,
    )

volume_builder.reconstruct(center_shift=0, chunk_count=5)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='att',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        window_size=umpa_w,
    )

volume_builder.reconstruct(center_shift=0, chunk_count=5)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='df_positive_stitched_processed',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        window_size=umpa_w
    )

volume_builder.reconstruct(center_shift=0, chunk_count=20)