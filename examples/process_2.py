from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.core.data_loader import DataLoader
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.core.volume_builder import VolumeBuilder
from tqdm import tqdm
import h5py
from monash_processing.utils.utils import Utils
import pyqtgraph as pg
from pathlib import Path
import numpy as np

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "P6_Manual"
pixel_size = 1.444e-6 # m
energy = 25 # keV
prop_distance = 0.32 # 17 cm sample-grid,
max_angle = 182
umpa_w = 1
n_workers = 50

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = DataLoader(scan_path, scan_name)
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

# 4. Phase integrate
print("Phase integrating")
#area_left, area_right = Utils.select_areas(loader.load_projections(projection_i=0)[0])
area_left = np.s_[100:-100, 20:120]
area_right = np.s_[100:-100, -120:-20]
parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader)
parallel_phase_integrator.integrate_parallel(num_angles, n_workers=n_workers)


##############################################################################################################
# 3D center shift finder
##############################################################################################################

#print('Find centershift')
#calibrator = ReconstructionCalibrator(loader)
#center_shift = calibrator.find_center_shift(
#    max_angle=max_angle,
#    num_projections=500,
#    test_range=(10, 30),
#    stepping=15,
#    pixel_size=pixel_size,
#    is_stitched=False
#)
energy = 25000
##############################################################################################################
center_shift = 24
volume_builder = VolumeBuilder(pixel_size, max_angle, 'phase', loader, center_shift=center_shift, energy=energy, is_stitched=False, method='FBP')
volume = volume_builder.reconstruct()
#pg.image(volume)

center_shift = 24
volume_builder = VolumeBuilder(pixel_size, max_angle, 'att', loader, center_shift=center_shift, energy=energy, is_stitched=False, method='FBP')
volume = volume_builder.reconstruct()

#### OR

volume_builder = VolumeBuilder(pixel_size, max_angle, 'phase', loader, center_shift, method='FBP', limit_max_angle=False)
volume = volume_builder.reconstruct_3d(enable_short_scan=True, debug=True)

pg.image(volume)