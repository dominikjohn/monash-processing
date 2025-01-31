from monash_processing.core.data_loader import DataLoader
import h5py
from pathlib import Path
import numpy as np
from monash_processing.core.volume_builder import VolumeBuilder
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher
import matplotlib
#matplotlib.use('TkAgg', force=True)

# Set your parameters
scan_path = Path("/vault3/other/AustralianSynchrotron/K3_3H_Manual/")
scan_name = "K3_3H_Manual"
pixel_size = 1.444e-6 # m
energy = 25000 # eV
prop_distance = 0.155 #
max_angle = 364
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

best_value = 1311
stitcher = ProjectionStitcher(loader, angle_step, center_shift=best_value / 2)
area_left = np.s_[:, 5:80]
area_right = np.s_[:, -80:-5]
stitcher.process_and_save_range(index_0, index_180, 'dx_')
stitcher.process_and_save_range(index_0, index_180, 'dy')
#stitcher.process_and_save_range(index_0, index_180, 'T')
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, stitched=True)
parallel_phase_integrator.integrate_parallel(max_index + 1, n_workers=n_workers)

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
    )

volume_builder.reconstruct(center_shift=0, chunk_count=20)