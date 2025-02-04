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
scan_name = "P6_Manual"
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

delta_ethanol = 2.957e-7
delta_pvc = 4.772e-7
delta_ptfe = 7.018e-7

mu_ethanol = 0.311 * 100
mu_pvc = 3.422 * 100
mu_ptfe = 1.264 * 100

for i in tqdm(range(index_0, index_180+1)):
    proj = (np.mean(loader.load_projections(i), axis=0) - dark_current)[slicer]
    A = loader.load_processed_projection(i, 'thickness')

    T_pvc = beltran_two_material_filter_modified(proj, flat_cor, mu_ethanol, mu_pvc, delta_ethanol, delta_pvc, pixel_size, prop_distance)
    T_ptfe = beltran_two_material_filter_modified(proj, flat_cor, mu_ethanol, mu_ptfe, delta_ethanol, delta_ptfe, pixel_size, prop_distance)

    loader.save_tiff('T_pvc_modified', i, T_pvc)
    loader.save_tiff('T_ptfe_modified', i, T_ptfe)

def process_projection(i, loader, dark_current, slicer, flat_cor,
                       mu_ethanol, mu_pvc, mu_ptfe,
                       delta_ethanol, delta_pvc, delta_ptfe,
                       pixel_size, prop_distance):
    # Load and process projection
    proj = (np.mean(loader.load_projections(i), axis=0) - dark_current)[slicer]
    A = loader.load_processed_projection(i, 'thickness')

    # Calculate thickness maps
    T_pvc = beltran_two_material_filter_modified(
        proj, flat_cor, mu_ethanol, mu_pvc,
        delta_ethanol, delta_pvc, pixel_size, prop_distance
    )
    T_ptfe = beltran_two_material_filter_modified(
        proj, flat_cor, mu_ethanol, mu_ptfe,
        delta_ethanol, delta_ptfe, pixel_size, prop_distance
    )

    # Save results
    loader.save_tiff('T_pvc_modified', i, T_pvc)
    loader.save_tiff('T_ptfe_modified', i, T_ptfe)

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
    mu_ethanol=mu_ethanol,
    mu_pvc=mu_pvc,
    mu_ptfe=mu_ptfe,
    delta_ethanol=delta_ethanol,
    delta_pvc=delta_pvc,
    delta_ptfe=delta_ptfe,
    pixel_size=pixel_size,
    prop_distance=prop_distance
)

# Create index list
indices = range(index_0, index_180)

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