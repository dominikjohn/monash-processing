from monash_processing.core.data_loader import DataLoader
from monash_processing.core.volume_builder import VolumeBuilder
from tqdm import tqdm
import h5py
import pyqtgraph as pg
from pathlib import Path
import numpy as np
import tifffile
from multiprocessing import Pool
from functools import partial


# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K3_4HE_Manual"
pixel_size = 1.444e-6  # m
energy = 25000
prop_distance = 0.155  # 17 cm sample-grid,
max_angle = 182
umpa_w = 1
n_workers = 50

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

# We take away a bit from the projections because the A image is based on UMPA and therefore cut smaller.
slicer = np.s_[5:-5, 5:-5]
flat_cor = (np.mean(flat_fields, axis=0) - dark_current)[slicer]

delta_st = 3.4265e-07
delta_pb = 3.0319e-06

mu_pb = 550.288 * 100 # 1/m
mu_st = .576 * 100 # 1/m

def beltran_two_material_filter_modified(I, I_ref, mu_enc, mu_2, delta_enc, delta_2, p_size, distance):
    '''
    Beltran et al. 2D and 3D X-ray phase retrieval of multi-material objects using a single defocus distance (2010)
    :param I: intensity projection with sample
    :param I_ref: reference projection
    :param A: total thickness map
    :param mu_enc: mu of enclosing material (e.g. soft tissue)
    :param mu_2: mu of enclosed material (e.g. contrast agent)
    :param delta_enc: delta of enclosing material (e.g. soft tissue)
    :param delta_2: delta of enclosed material (e.g. contrast agent)
    :param p_size: pixel size
    :param distance: propagation distance
    :return: Thickness map
    '''

    # Get image dimensions
    ny, nx = I.shape

    # Calculate frequencies using fftfreq
    delta_x = p_size / (2 * np.pi)
    kx = np.fft.fftfreq(nx, d=delta_x)
    ky = np.fft.fftfreq(ny, d=delta_x)

    # Create 2D frequency grid
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_squared = kx_grid ** 2 + ky_grid ** 2

    image_fft = np.fft.fft2(I / I_ref)

    denom = (distance * (delta_2 - delta_enc) / (mu_2 - mu_enc)) * k_squared + 1
    filter = 1 / denom

    filtered_fft = image_fft * filter
    log_image = -np.log(np.real(np.fft.ifft2(filtered_fft)))

    return log_image

for i in tqdm(range(index_0, index_180+1)):
    proj = (np.mean(loader.load_projections(i), axis=0) - dark_current)[slicer]
    T_lead_st = beltran_two_material_filter_modified(proj, flat_cor, mu_st, mu_pb, delta_st, delta_pb, pixel_size, prop_distance)
    loader.save_tiff('T_pb_st', i, T_lead_st)

def process_projection(i, loader, dark_current, slicer, flat_cor,
                       mu_enc, mu_inside,
                       delta_enc, delta_inside,
                       pixel_size, prop_distance):
    proj = (np.mean(loader.load_projections(i), axis=0) - dark_current)[slicer]

    T_pb_st = beltran_two_material_filter_modified(
        proj, flat_cor, mu_enc, mu_inside,
        delta_enc, delta_inside, pixel_size, prop_distance
    )
    loader.save_tiff('T_pb_st', i, T_pb_st)
    return i  # Return index for progress tracking

# Number of CPU cores to use (adjust as needed)
num_cores = 32

# Create partial function with all constant parameters
process_proj_partial = partial(
    process_projection,
    loader=loader,
    dark_current=dark_current,
    slicer=slicer,
    flat_cor=flat_cor,
    mu_enc=mu_st,
    mu_inside=mu_pb,
    delta_enc=delta_st,
    delta_inside=delta_pb,
    pixel_size=pixel_size,
    prop_distance=prop_distance
)

# Create index list
#indices = range(index_0, index_180+1)
indices = range(index_180, index_180*2+1)

# Create process pool and run parallel processing
with Pool(num_cores) as pool:
    # Use imap for progress tracking
    for _ in tqdm(pool.imap_unordered(process_proj_partial, indices),
                  total=len(indices)):
        pass

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=False,
        channel='T_ptfe_modified',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
    )

#center_shifts = np.linspace(10, 25, 10)
#volume_builder.sweep_centershift(center_shifts)

volume_builder.reconstruct(center_shift=17.5, chunk_count=10)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=False,
        channel='T_ptfe',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
    )

#volume_builder.sweep_centershift(np.linspace(17.5, 19, 5))
volume_builder.reconstruct(center_shift=17.5, chunk_count=10)