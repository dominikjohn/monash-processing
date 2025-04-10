from monash_processing.core.new_multi_position_data_loader import NewMultiPositionDataLoader
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
try:
    from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
except ImportError:
    UMPAProcessor = None
    print("Warning: UMPA not available.")
from monash_processing.postprocessing.stitch_phase_images import ProjectionStitcher
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path("/data/mct/22878/")
scan_name = "tumour_1_25_SBI_z20p5_4p5x_23keV_360CT"
pixel_size = 1.444e-6  # m
energy = 23000
prop_distance = 0.25
max_angle = 364
angle_count = 3640
umpa_w = 2
n_workers = 50

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = NewMultiPositionDataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

angles = loader.load_angles(max_angle=max_angle, angle_count=3640)
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

from monash_processing.core.volume_builder import VolumeBuilder

area_left = np.s_[:, 5:80]
area_right = np.s_[:, -80:-5]

max_index = int(np.round(180 / angle_step))
print('Uppermost projection index: ', max_index)

blending = False
center_shift_list = np.linspace(1200, 1400, 5)
for center_shift in center_shift_list:
    suffix = f'{(2 * center_shift):.2f}'
    stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=center_shift / 2, slices=(500, 510), suffix=suffix, window_size=umpa_w)
    stitcher.process_and_save_range(index_0, index_180, 'dx', blending=blending)
    stitcher.process_and_save_range(index_0, index_180, 'dy', blending=blending)
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
stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=best_value / 2, format='tif', window_size=umpa_w)
stitcher.process_and_save_range(index_0, index_180, 'dx', blending=blending)
stitcher.process_and_save_range(index_0, index_180, 'dy', blending=blending)
stitcher.process_and_save_range(index_0, index_180, 'T')
stitcher.process_and_save_range(index_0, index_180, 'df')

#stitcher = ProjectionStitcher(loader, angle_spacing=angle_step, center_shift=best_value / 2, format='tif', window_size=umpa_w)
#stitcher.process_and_save_range(index_0, index_180, 'df_positive')
area_left = np.s_[:, 5:80]
area_right = np.s_[:, -80:-5]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, window_size=umpa_w, stitched=True)
parallel_phase_integrator.integrate_parallel(max_index+1, n_workers=n_workers)

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