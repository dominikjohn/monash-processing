from monash_processing.core.data_loader_p10 import DataLoaderP10
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from pathlib import Path
import numpy as np
import hdf5plugin

# Set your parameters
scan_path = Path('/asap3/petra3/gpfs/p10/2025/data/11021161/')
scan_name = "placenta_04_tomo08"
pixel_size = 1.444e-6 # m
energy = 8000 # eV
prop_distance = 0.155
max_angle = 360
projection_count = 501
flat_count = 25
umpa_w = 1
n_workers = 100
cropping = np.s_[...]

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoaderP10(scan_path, scan_name, '20250405/detectors/eiger', flat_count=flat_count, projection_count=projection_count)
flat_fields = loader.load_flat_fields()
angles = np.linspace(0, 360, projection_count)

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
    umpa_w,
    n_workers=50,
    slicing=np.s_[..., 1100:1500, 300:800]
)

# Process projections
results = processor.process_projections(
    num_angles=projection_count,
)

area_left = np.s_[: 50:100]
area_right = np.s_[:, -50:-100]

parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader)
parallel_phase_integrator.integrate_parallel(projection_count, n_workers=n_workers)

from monash_processing.core.volume_builder import VolumeBuilder

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
        is_360_deg=True,
        window_size=umpa_w,
    )

center_shifts = np.linspace(10, 20, 40)
volume_builder.sweep_centershift(center_shifts)

volume_builder.reconstruct(center_shift=5.3, chunk_count=1)

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
    )

volume_builder.reconstruct(center_shift=0, chunk_count=20)
