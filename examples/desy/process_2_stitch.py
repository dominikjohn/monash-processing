from core.volume_builder_desy import VolumeBuilderDesy
from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from monash_processing.core.volume_builder import VolumeBuilder
from monash_processing.core.data_loader_desy import DataLoaderDesy
#from monash_processing.algorithms.umpa_wrapper import UMPAProcessor
#from monash_processing.core.volume_builder import VolumeBuilder
#from monash_processing.algorithms.parallel_phase_integrator import ParallelPhaseIntegrator
from pathlib import Path
import numpy as np
from monash_processing.postprocessing.stitch_phase_images_non_offset import ProjectionStitcherNonOffset

scan_base = '/asap3/petra3/gpfs/p07/2024/data/11020408/'
scan_name_left = "scratch_cc/016_basel5_a_left_dpc/w3/"
#scan_name_left = "processed/016_basel5_a_left_dpc/w3/"
#scan_name_right = "processed/016_basel5_a_right_dpc/w3/"
scan_name_right = "scratch_cc/016_basel5_a_right_dpc/w3/"

stitched_name = "processed/016_basel5_a_stitched_dpc/"
if not Path(scan_base + stitched_name).exists():
    Path(scan_base + stitched_name).mkdir()

pixel_size = 1.434e-6 # m
energy = 25000 # keV
prop_distance = 0.158 #
max_angle = 182
umpa_w = 1
n_workers = 50

max_index = 4501
angle_step = np.round(180/max_index)
loader_left = DataLoaderDesy(scan_base, scan_name_left)
loader_right = DataLoaderDesy(scan_base, scan_name_right)
loader_stitched = DataLoaderDesy(scan_base, stitched_name)

angles = np.linspace(0, 180, max_index)

area_left = np.s_[:, 50:600]
area_right = np.s_[:, -600:-50]


center_shift_list = np.linspace(1150, 1220, 15)
for center_shift in center_shift_list:
    suffix = f'{(2 * center_shift):.2f}'
    stitcher = ProjectionStitcherNonOffset(loader_left, loader_right, loader_stitched, angle_spacing=angle_step, center_shift=center_shift / 2, slices=(1000, 1020), suffix=suffix, format='tif')
    stitcher.process_and_save_range(0, max_index-1, 'dx')
    stitcher.process_and_save_range(0, max_index-1, 'dy')
    parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                        loader_stitched, stitched=True, suffix=suffix, simple_format=False, slicing=slicing)
    parallel_phase_integrator.integrate_parallel(max_index, n_workers=n_workers)
    volume_builder = VolumeBuilder(
        data_loader=loader_stitched,
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
        suffix=suffix
    )
    volume_builder.reconstruct(center_shift=0, chunk_count=1, custom_folder='offset_sweep', slice_range=(2,4))


best_value = 1185
center_shift = 0

#stitcher = ProjectionStitcherNonOffset(loader_left, loader_right, loader_stitched, angle_spacing=angle_step, center_shift=best_value / 2, format='tif')
#stitcher.process_and_save_range(0, max_index-1, 'dx')
#stitcher.process_and_save_range(0, max_index-1, 'dy')
#stitcher.process_and_save_range(0, max_index-1, 'T')
area_left = np.s_[:, 20:150]
area_right = np.s_[:, -150:-20]

slicing = np.s_[250:-250, 1000:-1000]

parallel_phase_integrator = ParallelPhaseIntegrator(energy, prop_distance, pixel_size, area_left, area_right,
                                                    loader_stitched, stitched=True, simple_format=False, slicing=slicing)
parallel_phase_integrator.integrate_parallel(max_index, n_workers=n_workers)

volume_builder = VolumeBuilderDesy(
        data_loader=loader_stitched,
        original_angles=angles,
        energy=energy,
        prop_distance=prop_distance,
        pixel_size=pixel_size,
        is_stitched=True,
        channel='original_phi',
        detector_tilt_deg=0,
        show_geometry=False,
        sparse_factor=1,
        is_360_deg=False,
    )
volume_builder.reconstruct(center_shift=center_shift, chunk_count=100)



