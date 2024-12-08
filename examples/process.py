#from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
import os

from monash_processing.core.data_loader import DataLoader
#from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
#from monash_processing.core.volume_builder import VolumeBuilder
#from monash_processing.algorithms.centershift_finder import ReconstructionCalibrator
from tqdm import tqdm
import h5py
from monash_processing.utils.utils import Utils
import pyqtgraph as pg
from pathlib import Path
import numpy as np
import tifffile

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "P6_ReverseOrder"
pixel_size = 1.434e-6 # m
energy = 25000 # keV
prop_distance = 0.158 #
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
with processor:
    results = processor.process_projections(
        flats=flat_fields,
        num_angles=180  # Or however many angles you have
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

print('Find centershift')
calibrator = ReconstructionCalibrator(loader)
center_shift = calibrator.find_center_shift(
    max_angle=max_angle,
    slice_idx=1,
    num_projections=500,
    test_range=(53, 56),
    pixel_size=pixel_size,
    is_stitched=False
)

##############################################################################################################
center_shift = 54
volume_builder = VolumeBuilder(pixel_size, max_angle, 'phase', loader, center_shift=center_shift, energy=energy, is_stitched=False, method='FBP')
volume = volume_builder.reconstruct()
#pg.image(volume)

center_shift = 54
volume_builder = VolumeBuilder(pixel_size, max_angle, 'att', loader, center_shift=center_shift, energy=energy, is_stitched=False, method='FBP')
volume = volume_builder.reconstruct()

#### OR

volume_builder = VolumeBuilder(pixel_size, max_angle, 'phase', loader, center_shift, method='FBP', limit_max_angle=False)
volume = volume_builder.reconstruct_3d(enable_short_scan=True, debug=True)



##### EXPERIMENTAL
from cil.framework import AcquisitionGeometry
from cil.utilities.display import show_geometry
from cil.framework import AcquisitionData
from cil.processors import RingRemover
from cil.recon import FBP

def load_projections(is_stitched, format='tif', channel='phase'):
    """
    :return: np.ndarray, np.ndarray
    """
    if is_stitched:
        input_dir = loader.results_dir / ('phi_stitched' if channel == 'phase' else 'T_stitched')
    else:
        input_dir = loader.results_dir / ('phi' if channel == 'phase' else 'T')

    tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))

    # Generate angles and create mask for <= 180°
    angles = np.linspace(0, max_angle, len(tiff_files))

    valid_indices = np.arange(len(tiff_files))

    projections = []
    for projection_i in tqdm(valid_indices, desc=f"Loading {channel} projections", unit="file"):
        try:
            data = tifffile.imread(tiff_files[projection_i])
            projections.append(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

    return np.array(projections), angles

projections, angles = load_projections(is_stitched=False)

scaling_factor = 1e3
source_distance = 21.5 * scaling_factor
detector_distance = prop_distance * scaling_factor
pix_size_scaled = pixel_size * scaling_factor

detector_tilt_deg = 0
detector_tilt = np.radians(detector_tilt_deg)

angles_reduced = angles[0:1800]
projection_shape = projections.shape
chunk_size = 100

for i in range(projection_shape[1]//chunk_size):
    projections_reduced = projections[0:1800, i*chunk_size:(i+1)*chunk_size, :]

    n_rows = projections_reduced.shape[1]
    n_cols = projections_reduced.shape[2]
    center_shift = 38.75
    rot_offset_pix = -center_shift * scaling_factor

    source_position = [0,-source_distance,0]
    detector_position = [0,detector_distance,0]

    rot_axis_shift = rot_offset_pix * pix_size_scaled

    #detector_direction_x = [np.cos(detector_tilt), 0, np.sin(detector_tilt)]
    #detector_direction_y = [-np.sin(detector_tilt), 0, np.cos(detector_tilt)]
    detector_direction_y = [0,0,1]
    detector_direction_x = [np.cos(detector_tilt), np.sin(detector_tilt), 0]

    ag = AcquisitionGeometry.create_Parallel3D(
        detector_position=detector_position,
        detector_direction_x=detector_direction_x,
        detector_direction_y=detector_direction_y,
        rotation_axis_position=[rot_axis_shift, 0, 0])\
        .set_panel(num_pixels=[n_cols, n_rows])\
        .set_angles(angles=angles_reduced)

    #show_geometry(ag)
    data = AcquisitionData(projections_reduced.astype('float32'), geometry=ag)

    ring_filter = RingRemover()
    ring_filter.set_input(data)
    data = ring_filter.get_output()

    fdk = FBP(data)
    #fdk.set_splits(5)
    out = fdk.run()

    save_folder = str(scan_path / 'results' / scan_name / 'recon')
    os.makedirs(save_folder, exist_ok=True)
    writer = cil.io.TIFFWriter(out, save_folder + '/recon', counter_offset=i*chunk_size)
    writer.write()