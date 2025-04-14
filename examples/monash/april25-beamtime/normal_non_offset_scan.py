from monash_processing.core.new_multi_position_data_loader import NewMultiPositionDataLoader
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
import h5py
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path("/data/mct/22878/")
scan_name = "carotidartery_clot_SBI_z20p5_4p5x_23keV_CT_3"
pixel_size = 1.444e-6  # m
energy = 23000
prop_distance = 0.25  # 17 cm sample-grid,
max_angle = 182
angle_count = 1820
umpa_w = 2
n_workers = 50

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = NewMultiPositionDataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

angles = loader.load_angles(max_angle=max_angle, angle_count=angle_count)
angle_step = np.diff(angles).mean()
print('Angle step:', angle_step)
index_0 = np.argmin(np.abs(angles - 0))
index_180 = np.argmin(np.abs(angles - 180))
print('Index at 0°:', index_0)
print('Index at 180°:', index_180)

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
    n_workers=50,
    w=umpa_w,
    slicing=np.s_[..., :, :]
)

# Process projections
results = processor.process_projections(
    num_angles=angle_count,
)

# 4. Phase integrate
print("Phase integrating")
# area_left, area_right = Utils.select_areas(loader.load_projections(projection_i=0)[0])
area_left = np.s_[100:-100, 20:120]
area_right = np.s_[100:-100, -120:-20]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader, window_size=umpa_w)
parallel_phase_integrator.integrate_parallel(index_180, n_workers=20)

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
        is_360_deg=False,
        readjust_angles=True,
        window_size=umpa_w,
    )

volume_builder.sweep_centershift(np.linspace(-1, 0, 10))
volume_builder.reconstruct(center_shift=-1, chunk_count=10)

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

#volume_builder.sweep_centershift(np.linspace(17.5, 19, 5))
volume_builder.reconstruct(center_shift=17.5, chunk_count=10)