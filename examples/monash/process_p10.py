from monash_processing.core.data_loader_p10 import DataLoaderP10
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.core.volume_builder import VolumeBuilder
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path('/asap3/petra3/gpfs/p10/2025/data/11021161/')
scan_name = "placenta_04_tomo_08"
pixel_size = 1.444e-6 # m
energy = 8000 # eV
prop_distance = 0.155
max_angle = 360
projection_count = 500
flat_count = 25
umpa_w = 1
n_workers = 100
cropping = np.s_[...]

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoaderP10(scan_path, scan_name, '20250406/detectors/eiger', flat_count=flat_count)
flat_fields = loader.load_flat_fields()
angles = np.linspace(0, 360, projection_count)
print(angles)

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
    num_angles=projection_count,
)

area_left = np.s_[: 5:80]
area_right = np.s_[:, -80:-5]

parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader, stitched=True)
parallel_phase_integrator.integrate_parallel(projection_count+1, n_workers=n_workers)

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
