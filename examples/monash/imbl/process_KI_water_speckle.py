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
    n_workers=50
)

# Process projections
results = processor.process_projections(
    num_angles=num_angles
)

#center_shifts = np.linspace(307, 312, 10)
#volume_builder.sweep_centershift(center_shifts)
area_left = np.s_[:-650, 5:150]
area_right = np.s_[:-650, -150:-5]

parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader, stitched=False)
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
    )

volume_builder.reconstruct(center_shift=0, chunk_count=20)

import re
import numpy as np
from typing import List, Dict, Tuple


def extract_sample_angles(log_text: str) -> np.ndarray:
    """
    Extract angles from sample scans and return as 2D numpy array [n_scans, n_angles].
    Handles gaps in indices by filling a fixed-size array.
    """
    scans: Dict[str, Dict[int, float]] = {}
    current_scan: Dict[int, float] = {}
    is_sample = False
    scan_name = ""
    max_index = 0

    for line in log_text.split('\n'):
        if 'filename prefix' in line and 'SAMPLE_' in line:
            if current_scan and scan_name:
                scans[scan_name] = current_scan
            current_scan = {}
            is_sample = True
            scan_name = re.search(r'"([^"]+)"', line).group(1)
        elif 'Acquisition finished' in line:
            if is_sample and current_scan and scan_name:
                scans[scan_name] = current_scan
            current_scan = {}
            is_sample = False
        elif is_sample:
            # Extract both index and angle
            match = re.match(r'\d{4}-\d{2}-\d{2}.*?(\d+)\s+(\d+\.\d+)', line)
            if match:
                idx, angle = int(match.group(1)), float(match.group(2))
                current_scan[idx] = angle
                max_index = max(max_index, idx)

    # Print diagnostic information
    print(f"Found {len(scans)} sample scans:")
    for name, measurements in scans.items():
        indices = list(measurements.keys())
        print(f"  {name}: {len(measurements)} measurements, indices: {min(indices)}-{max(indices)}")

    # Create array and fill values
    array_size = max_index + 1
    angle_array = np.full((len(scans), array_size), np.nan)

    for i, (name, measurements) in enumerate(sorted(scans.items())):
        for idx, angle in measurements.items():
            angle_array[i, idx] = angle

    print(f"\nFinal array shape: {angle_array.shape}")
    print("Used scans (in order):")
    for name in sorted(scans.keys()):
        print(f"  {name}")

    # Print statistics about gaps
    nan_counts = np.isnan(angle_array).sum(axis=1)
    for i, (name, nan_count) in enumerate(zip(sorted(scans.keys()), nan_counts)):
        gap_percent = (nan_count / array_size) * 100
        print(f"  {name}: {nan_count} gaps ({gap_percent:.1f}%)")

    return angle_array

with open('/data/imbl/23081/input/Day3/Dominik_KI_salts_0p75m_30keV_0p16s/acquisition.0.log', 'r') as f:
    angles = np.nanmean(extract_sample_angles(f.read()), axis=0)


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