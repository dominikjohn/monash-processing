from monash_processing.core.data_loader import ScanDataLoader
from monash_processing.core.preprocessor import ImagePreprocessor
from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
from monash_processing.core.volume_builder import VolumeBuilder
from tqdm import tqdm
import h5py

# Set your parameters
scan_path = "/path/to/scan"
scan_name = "scan_123"

# 1. Load reference data
print(f"Loading data from {scan_path}, scan name: {scan_name}")
loader = ScanDataLoader(scan_path, scan_name)
flat_fields = loader.load_flat_fields()
dark_current = loader.load_dark_current()

# Get number of projections (we need this for the loop)
with h5py.File(loader.h5_files[0], 'r') as f:
    num_angles = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA'].shape[0]

# 2. Initialize preprocessor and UMPA processor
print("Initializing processors")
umpa_processor = UMPAProcessor(scan_path, scan_name)

# 3. Process each projection
print("Processing projections")
for angle_idx in tqdm(range(num_angles), desc="Processing projections"):
    # Load single projection
    projection = loader.load_projections(projection_idx=angle_idx)

    # Preprocess
    corrected_projection = preprocessor.apply_corrections(projection)

    # Apply UMPA
    umpa_results = umpa_processor.process_projection(
        flat_fields,
        corrected_projection,
        angle_idx
    )

# 4. Load results when needed
results_dir = umpa_processor.results_dir
full_results = umpa_processor.load_results(results_dir, num_angles)

# 5. Reconstruct volume (when ready)
print("Reconstructing volume")
volume_builder = VolumeBuilder()
volume = volume_builder.reconstruct(full_results)

# Now you have all variables in your interactive namespace
# and can inspect/modify them as needed