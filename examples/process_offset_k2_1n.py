from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.core.multi_position_data_loader import MultiPositionDataLoader
#from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.core.volume_builder import VolumeBuilder
import h5py
from pathlib import Path
import numpy as np
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2

import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K21N_sample"
pixel_size = 1.444e-6 # m
energy = 25000 # eV
prop_distance = 0.15 # Grid to detector in this case, still not sure why
max_angle = 364
umpa_w = 1
n_workers = 100

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = MultiPositionDataLoader(scan_path, scan_name, skip_positions={'03_03'})
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

area_left = np.s_[: 5:80]
area_right = np.s_[:, -80:-5]

center_shift_list = np.arange(700, 1000, 50)
for center_shift in center_shift_list:
    suffix = f'{(2 * center_shift):.2f}'
    stitcher = ProjectionStitcher(loader, .1, center_shift=center_shift / 2, slices=(1000, 1020), suffix=suffix)
    stitcher.process_and_save_range(0, 1799, 'dx')
    stitcher.process_and_save_range(0, 1799, 'dy')
    parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                        loader, stitched=True, suffix=suffix)
    parallel_phase_integrator.integrate_parallel(1800, n_workers=n_workers)
    volume_builder = VolumeBuilder(
        data_loader=loader,
        max_angle=180,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='phase',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        is_offset=True,
        suffix=suffix
    )
    volume_builder.reconstruct(center_shift=0, chunk_count=1, custom_folder='offset_sweep', slice_range=(2,4))


best_value = 895
stitcher = ProjectionStitcher(loader, .1, center_shift=center_shift / 2)
stitcher.process_and_save_range(0, 1799, 'dx')
stitcher.process_and_save_range(0, 1799, 'dy')
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, stitched=True)
parallel_phase_integrator.integrate_parallel(1800, n_workers=n_workers)

volume_builder = VolumeBuilder(
        data_loader=loader,
        max_angle=180,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='phase',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
        is_offset=True,
    )
center_shift = 0
volume_builder.reconstruct(center_shift=center_shift, chunk_count=30)