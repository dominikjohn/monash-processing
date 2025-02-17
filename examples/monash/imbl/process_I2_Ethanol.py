from monash_processing.core.data_loader_imbl import IMBLDataLoader
from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
try:
    from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
except ImportError:
    from monash_processing.core.volume_builder import VolumeBuilder
import h5py
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt

# Set your parametersDominik_KI_water_speckle
scan_path = Path("/data/imbl/23081/input/Day3/")
scan_name = "Dominik_K3_1N_I2_25keV_1m"
pixel_size = 9.07e-6 # m
energy = 25000 # eV
prop_distance = 1.0 #
max_angle = 364
umpa_w = 1
n_workers = 100

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = IMBLDataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

angles = loader.load_angles()
angle_step = np.diff(angles).mean()
print('Angle step:', angle_step)
index_0 = np.argmin(np.abs(angles - 0))
index_360 = np.argmin(np.abs(angles - 360))
print('Index at 0°:', index_0)
print('Index at 360°:', index_360)
num_angles = angles.shape[0]

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
    n_workers=100
)

# Process projections
results = processor.process_projections(
    num_angles=num_angles
)

area_left = np.s_[:-1500, :120]
area_right = np.s_[:-1500, -35:]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, stitched=False)
parallel_phase_integrator.integrate_parallel(index_360+1, n_workers=n_workers)

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
        is_360_deg=True,
    )

volume_builder.sweep_centershift(np.linspace(-40, 40, 5))

volume_builder.reconstruct(center_shift=0, chunk_count=20)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='T_pb_st_stitched',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
    )

volume_builder.reconstruct(center_shift=0, chunk_count=20)

volume_builder.sweep_centershift([-1, 0.5, 0, 0.5, 1])

volume_builder.reconstruct(center_shift=0, chunk_count=20)