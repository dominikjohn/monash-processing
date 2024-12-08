import numpy as np
from tqdm import tqdm
import tifffile
import astra

from monash_processing.postprocessing.filters import RingFilter
from monash_processing.utils.utils import Utils
import scipy.constants
from cil.framework import AcquisitionGeometry
from cil.utilities.display import show_geometry
from cil.framework import AcquisitionData
from cil.processors import RingRemover
from cil.recon import FBP
import cil.io
import os
import tomopy

class VolumeBuilder:
    def __init__(self, data_loader, max_angle, energy, prop_distance, pixel_size, is_stitched=False, channel='phase', detector_tilt_deg=0, show_geometry=False):
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.max_angle_rad = np.deg2rad(max_angle)
        self.energy = energy
        self.prop_distance = prop_distance
        self.is_stitched = is_stitched
        self.detector_tilt_deg = detector_tilt_deg
        if detector_tilt_deg != 0:
            raise NotImplementedError("Detector tilt is not implemented yet")
        self.scaling_factor = 1e3
        self.source_distance = 21.5 * self.scaling_factor
        self.detector_distance = self.prop_distance * self.scaling_factor
        self.pix_size_scaled = self.pixel_size * self.scaling_factor
        self.show_geometry = show_geometry

        self.projections, self.angles = self.load_projections()

    def load_projections(self, format='tif'):
        """
        :return: np.ndarray, np.ndarray
        """
        if self.is_stitched:
            input_dir = self.data_loader.results_dir / ('phi_stitched' if self.channel == 'phase' else 'T_stitched')
        else:
            input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'T')

        tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))
        angles, valid_indices = self.get_valid_indices(self.max_angle_rad, len(tiff_files))

        projections = []
        for projection_i in tqdm(valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[projection_i])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), angles

    def get_valid_indices(self, max_angle, file_count):
        angles = np.linspace(0, self.max_angle_rad, file_count)
        valid_angles_mask = angles <= np.pi
        valid_indices = np.where(valid_angles_mask)[0]
        return angles[valid_angles_mask], valid_indices

    def get_acquisition_geometry(self, n_cols, n_rows, angles, center_shift):
        #source_position = [0, -self.source_distance, 0] # Not required for parallel beam
        detector_position = [0, self.detector_distance, 0]

        # Calculate displacements of rotation axis in pixels
        rot_offset_pix = -center_shift * self.scaling_factor
        rot_axis_shift = rot_offset_pix * self.pix_size_scaled

        detector_direction_y = [0, 0, 1]
        detector_direction_x = [1, 0, 0]

        ag = AcquisitionGeometry.create_Parallel3D(
            detector_position=detector_position,
            detector_direction_x=detector_direction_x,
            detector_direction_y=detector_direction_y,
            rotation_axis_position=[rot_axis_shift, 0, 0]) \
            .set_panel(num_pixels=[n_cols, n_rows]) \
            .set_angles(angles=angles)

        if self.show_geometry:
            show_geometry(ag)

        return ag

    def process_chunk(self, chunk_projections, angles, center_shift):
        n_rows = chunk_projections.shape[1]
        n_cols = chunk_projections.shape[2]
        ag = self.get_acquisition_geometry(n_cols, n_rows, angles, center_shift)

        data = AcquisitionData(chunk_projections.astype('float32'), geometry=ag)
        data = self.apply_projection_ring_filter(data)

        fdk = FBP(data)
        out = fdk.run()

        out = self.apply_reconstruction_ring_filter(out)

        return out

    def apply_projection_ring_filter(self, data):
        ring_filter = RingRemover()
        ring_filter.set_input(data)
        return ring_filter.get_output()

    def apply_reconstruction_ring_filter(self, data, rwidth=None):
        if rwidth is not None:
            return tomopy.misc.corr.remove_ring(data, rwidth=rwidth)
        else:
            return tomopy.misc.corr.remove_ring(data)

    def save_reconstruction(self, data, counter_offset, prefix='recon'):
        save_folder = self.data_loader.get_save_path() / prefix
        os.makedirs(save_folder, exist_ok=True)
        writer = cil.io.TIFFWriter(data, save_folder + f'/recon', counter_offset=counter_offset)
        writer.write()

    def calculate_beer_lambert(self, projections):
        epsilon = 1e-8
        projections = np.clip(projections, epsilon, 1.0)
        projections = -np.log(projections)
        return projections

    def convert_to_edensity(self, slice_result):
        wavevec = 2 * np.pi * self.energy / (
                scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
        r0 = scipy.constants.physical_constants['classical electron radius'][0]
        conv = wavevec / (2 * np.pi) / r0 / 1E27
        slice_result *= conv * 1E9  # convert to nm^3

        return slice_result

    def convert_to_mu(self, slice_result):
        slice_result *= self.scaling_factor  # corrects for the scaling factor
        slice_result /= 100  # converts to cm^-1
        return slice_result

    def reconstruct(self, center_shift=0, chunk_size=1, sparse_factor=1, custom_folder=None):
        if self.channel == 'att':
            # For attenuation images, we calculate the log first according to Beer-Lambert
            projections = self.calculate_beer_lambert(self.projections)

        n_slices = projections.shape[1]
        for i in range(n_slices // chunk_size):
            chunk_projections = projections[::sparse_factor, i * chunk_size:(i + 1) * chunk_size, :]
            volume = self.process_chunk(chunk_projections, self.angles, center_shift)
            rwidth = None

            if self.channel == 'phase':
                volume = self.convert_to_edensity(volume)
            elif self.channel == 'att':
                volume = self.convert_to_mu(volume)
                rwidth = 15 # Attenuation needs a larger ring filter width

            volume = self.apply_reconstruction_ring_filter(volume, rwidth=rwidth)
            prefix = 'recon' if custom_folder is None else custom_folder
            self.save_reconstruction(volume, counter_offset=i * chunk_size, prefix=prefix)

    def sweep_centershift(self, center_shift_range, chunk_size=1, sparse_factor=1):
        for center_shift in center_shift_range:
            print(f"Processing center shift {center_shift}")
            self.reconstruct(center_shift, chunk_size, sparse_factor, custom_folder='centershift_sweep')