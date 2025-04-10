from monash_processing.core.volume_builder import VolumeBuilder
from monash_processing.core.data_loader import DataLoader
#from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
#from monash_processing.core.volume_builder import VolumeBuilder
#from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher

import h5py
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "P6_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25000 # keV
prop_distance = 0.158 #
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
with processor:
    results = processor.process_projections(
        flats=flat_fields,
        num_angles=180
    )

# 4. Phase integrate
print("Phase integrating")
#area_left, area_right = Utils.select_areas(loader.load_projections(projection_i=0)[0])
area_left = np.s_[100:-100, 20:120]
area_right = np.s_[100:-100, -120:-20]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader)
parallel_phase_integrator.integrate_parallel(num_angles, n_workers=n_workers)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=False,
        channel='phase',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        readjust_angles=True,
    )

volume_builder.sweep_centershift(np.linspace(38, 45, 5))

sparse_factor = 1 # Only use every n-th projection
#center_shifts = np.linspace(30, 70, 5)
#volume_builder.sweep_centershift(center_shifts)

center_shift = 38.8
volume_builder.reconstruct(center_shift=center_shift, chunk_count=30)

volume_builder = VolumeBuilder(
        data_loader=loader,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=False,
        channel='att',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        readjust_angles=True,
    )
volume_builder.reconstruct(center_shift=center_shift, chunk_count=10)
