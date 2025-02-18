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
scan_name = "Dominik_KI_salts_0p75m_30keV_0p16s"
pixel_size = 9.07e-6 # m
energy = 30000 # eV
prop_distance = .75 #
max_angle = 364
umpa_w = 15
n_workers = 100

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = IMBLDataLoader(scan_path, scan_name)
angles = loader.load_angles()

flat_fields = loader.load_flat_fields()
dark_current = loader.load_flat_fields(dark=True)

angle_step = np.diff(angles).mean()
print('Angle step:', angle_step)
index_0 = np.argmin(np.abs(angles - 0))
index_360 = np.argmin(np.abs(angles - 360))
print('Index at 0°:', index_0)
print('Index at 360°:', index_360)
num_angles = angles.shape[0]

slicing = np.s_[..., 300:370, :]

# 2. Initialize UMPA processor
processor = UMPAProcessor(
    scan_path,
    scan_name,
    loader,
    umpa_w,
    n_workers=50,
    slicing=slicing
)

# Process projections
results = processor.process_projections(
    num_angles=num_angles
)

#center_shifts = np.linspace(307, 312, 10)
#volume_builder.sweep_centershift(center_shifts)
#area_left = np.s_[:-650, 5:150]
area_left = np.s_[:, 5:150]
#area_right = np.s_[:-650, -150:-5]
area_right = np.s_[:, -150:-5]

parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, window_size=15, stitched=False, slicing=None)
parallel_phase_integrator.integrate_parallel(2000, n_workers=n_workers)

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
        window_size=umpa_w
    )

volume_builder.reconstruct(center_shift=0, chunk_count=2)


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
        is_360_deg=True,
    )

volume_builder.sweep_centershift(np.linspace(-2, 2, 5))

volume_builder.reconstruct(center_shift=0, chunk_count=20)