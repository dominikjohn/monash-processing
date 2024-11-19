from monash_processing.core.data_loader import DataLoader
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.algorithms.phase_integration import PhaseIntegrator
from monash_processing.core.volume_builder import VolumeBuilder
from tqdm import tqdm
import h5py
from utils.utils import Utils
import pyqtgraph as pg
from pathlib import Path

# Set your parameters
scan_path = Path("/path/to/scan")
scan_name = "P6_ReverseOrder"
pixel_size = 1.444e-6 # m
energy = 25 # keV
prop_distance = 0.158 #
max_angle = 382

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
umpa_processor = UMPAProcessor(scan_path, scan_name)

# 3. Process each projection
print("Processing projections")
for angle_i in tqdm(range(num_angles), desc="Processing projections"):
    # Load single projection
    projection = loader.load_projections(projection_i=angle_i)

    umpa_processor.process_projection(
        flat_fields,
        projection,
        angle_i)

# 4. Phase integrate
print("Phase integrating")
area_left, area_right = Utils.select_areas(loader.load_projections(projection_i=0)[0])
phase_integrator = PhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right, loader)

for angle_i in tqdm(range(num_angles), desc="Integrating projections"):
    phase_integrator.integrate_single(angle_i)

# 5. Reconstruct volume
print("Reconstructing volume")
volume_builder = VolumeBuilder()
volume = volume_builder.reconstruct()

pg.image(volume)

# Now you have all variables in your interactive namespace
# and can inspect/modify them as needed