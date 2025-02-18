import numpy as np
from tqdm import tqdm
import tifffile

import scipy.constants
from cil.framework import AcquisitionGeometry
from cil.utilities.display import show_geometry
from cil.framework import AcquisitionData
from cil.framework import ImageData
from cil.processors import RingRemover
from cil.recon import FBP
import cil.io
import os
from monash_processing.postprocessing.binner import Binner

class VolumeBuilder:
    def __init__(self, data_loader, original_angles, energy, prop_distance, pixel_size, is_stitched=False, channel='phase',
                 detector_tilt_deg=0, show_geometry=False, sparse_factor=1, is_360_deg=False, suffix=None, window_size=None, readjust_angles=False):
        self.data_loader = data_loader
        self.channel = channel
        self.pixel_size = pixel_size
        self.original_angles = original_angles
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
        self.is_360_deg = is_360_deg
        self.suffix = suffix
        self.window_size = window_size
        self.readjust_angles = readjust_angles
        print('Window size:', self.window_size)
        self.projections, self.angles = self.load_projections(sparse_factor=sparse_factor)

    def load_projections(self, sparse_factor=1, format='tif'):
        """
        Load projections with option to skip files based on sparse_factor.

        :param sparse_factor: int, load every nth projection (e.g., 2 means load every other projection)
        :param debug: bool, whether to return dummy data for debugging
        :param format: str, file format extension
        :return: np.ndarray, np.ndarray of projections and corresponding angles
        """

        base_path = self.data_loader.results_dir
        if self.window_size is not None:
            base_path = base_path / f'umpa_window{self.window_size}'

        if self.is_stitched:
            if self.suffix is not None:
                input_dir = base_path / (f'phi_stitched_{self.suffix}' if self.channel == 'phase' else 'T_stitched')
            else:
                input_dir = base_path / ('phi_stitched' if self.channel == 'phase' else 'T_stitched')
        else:
            if self.suffix is not None:
                input_dir = base_path/ (f'phi_{self.suffix}' if self.channel == 'phase' else 'T')
            else:
                input_dir = base_path / ('phi' if self.channel == 'phase' else 'T')

        if self.channel != 'phase' and self.channel != 'att':
            input_dir = base_path / self.channel
            print('Using custom input dir for channel:', self.channel)

        tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))
        print(tiff_files)
        angles, valid_indices = self.get_valid_indices(len(tiff_files))

        print('Angles:', angles)
        print('Valid indices:', valid_indices)

        # Apply sparse factor to valid indices
        sparse_valid_indices = valid_indices[::sparse_factor]
        sparse_angles = angles[::sparse_factor]

        projections = []
        for projection_i in tqdm(sparse_valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                data = tifffile.imread(tiff_files[projection_i])
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), sparse_angles

    def get_valid_indices(self, file_count):
        print(f"File count: {file_count}")
        print(f"Original angles: {self.original_angles}")

        angles = self.original_angles[:file_count].copy()
        indices = np.arange(file_count)

        if self.readjust_angles:
            # Subtract first angle from all angles to start at 0
            first_angle = angles[0]
            angles = angles - first_angle
            print(f"Readjusted angles to start at 0 (subtracted {first_angle})")

            # Filter angles based on scan type
            max_angle = 360.0 if self.is_360_deg else 180.0
            valid_mask = angles <= max_angle
            angles = angles[valid_mask]
            indices = indices[valid_mask]
            print(f"Filtered angles > {max_angle}°, {len(valid_mask) - np.sum(valid_mask)} angles removed")

        # Check final angle for warning
        final_angle = angles[-1]
        if self.is_360_deg:
            print(f"Actual angle 360: {final_angle}")
            if final_angle - 360 > 0.1:
                print(str(final_angle - 360))
                print(
                    '###### WARNING! The 360° projection is not within 0.1 of 360 degrees! This can lead to unexpected results.')
        else:
            print(f"Actual angle 180: {final_angle}")
            if final_angle - 180 > 0.1:
                print(str(final_angle - 180))
                print(
                    '###### WARNING! The 180° projection is not within 0.1 of 180 degrees! This can lead to unexpected results.')

        return angles, indices

    def get_acquisition_geometry(self, n_cols, n_rows, angles, center_shift):
        # source_position = [0, -self.source_distance, 0] # Not required for parallel beam
        detector_position = [0, self.detector_distance, 0]
        # Calculate displacements of rotation axis in pixels
        rot_offset_pix = -center_shift * self.scaling_factor
        rot_axis_shift = rot_offset_pix * self.pix_size_scaled

        #detector_direction_y = [0, 0, 1]
        #detector_direction_x = [1, 0, 0]


        ag = AcquisitionGeometry.create_Parallel3D(
            detector_position=detector_position,
            #detector_direction_x=detector_direction_x,
            #detector_direction_y=detector_direction_y,
            rotation_axis_position=[rot_axis_shift, 0, 0]) \
            .set_panel(num_pixels=[n_cols, n_rows]) \
            .set_angles(angles=angles)

        if self.show_geometry:
            show_geometry(ag)

        return ag

    def get_image_geometry(self, ag):
        ig = ag.get_ImageGeometry()
        return ig

    def process_chunk(self, chunk_projections, angles, center_shift):
        n_rows = chunk_projections.shape[1]
        n_cols = chunk_projections.shape[2]
        ag = self.get_acquisition_geometry(n_cols, n_rows, angles, center_shift)
        ig = self.get_image_geometry(ag)

        data = AcquisitionData(chunk_projections.astype('float32'), geometry=ag)
        #data = self.apply_projection_ring_filter(data)
        data.reorder('astra')
        fdk = FBP(data, image_geometry=ig, backend='astra')
        out = fdk.run()

        return out

    def apply_projection_ring_filter(self, data):
        ring_filter = RingRemover()
        ring_filter.set_input(data)
        return ring_filter.get_output()

    def apply_reconstruction_ring_filter(self, volume, geometry, rwidth=None):
        import tomopy
        data = volume.as_array()
        if rwidth is not None:
            data = tomopy.misc.corr.remove_ring(data, rwidth=rwidth)
        else:
            data = tomopy.misc.corr.remove_ring(data)
        return ImageData(data, geometry=geometry)

    def save_reconstruction(self, data, counter_offset, center_shift, prefix='recon'):
        save_folder = self.data_loader.get_save_path() / (prefix + '_' + self.channel)
        os.makedirs(save_folder, exist_ok=True)
        cs_formatted = self.get_shift_filename(center_shift)  # Center shift formatted to non-negative integer
        if self.suffix is not None:
            save_name = str(save_folder / f'recon_cs{cs_formatted}_{self.suffix}')
        else:
            save_name = str(save_folder / f'recon_cs{cs_formatted}')
        writer = cil.io.TIFFWriter(data, file_name = save_name, counter_offset=counter_offset)
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
        slice_result /= self.scaling_factor
        return slice_result

    def convert_to_mu(self, slice_result):
        slice_result *= self.scaling_factor**2  # corrects for the scaling factor
        slice_result /= 100  # converts to cm^-1
        return slice_result

    def reconstruct(self, center_shift=0, chunk_count=1, custom_folder=None, slice_range=None, binning_factor=1):
        """
        Reconstruct volume from projections with optional binning.

        Args:
            center_shift: Shift of rotation center
            chunk_count: Number of chunks to process
            custom_folder: Custom output folder name
            slice_range: Range of slices to process
            binning_factor: Factor by which to bin projections (default=1, no binning)
        """
        if slice_range is not None:
            projections = self.projections[:, slice_range, :]
        else:
            projections = self.projections

        if binning_factor > 1:
            print(f"\nBinning projections {binning_factor}x...")
            # Create a binner instance (path doesn't matter since we're only using _bin_2d)
            binner = Binner(".")

            # Bin each projection
            binned_projections = np.stack([
                binner._bin_2d(proj, binning_factor)
                for proj in tqdm(projections)
            ])

            projections = binned_projections
            print(f"Binned projection shape: {projections.shape}")

        if self.channel == 'att':
            # For attenuation images, we calculate the log first according to Beer-Lambert
            projections = self.calculate_beer_lambert(projections)

        n_slices = projections.shape[1]
        chunk_size = n_slices // chunk_count
        print('Chunk size:', chunk_size)

        for i in range(chunk_count):
            print(f"Processing chunk {i + 1}/{chunk_count}")
            chunk_projections = projections[:, i * chunk_size:(i + 1) * chunk_size, :]
            print(chunk_projections.shape)

            volume = self.process_chunk(chunk_projections, self.angles, center_shift)
            rwidth = None

            if self.channel == 'phase':
                volume = self.convert_to_edensity(volume)
                rwidth = 15

            elif self.channel == 'att' or self.channel == 'T_raw_stitched':
                volume = self.convert_to_mu(volume)
                rwidth = 15  # Attenuation needs a larger ring filter width

            volume = self.apply_reconstruction_ring_filter(volume, rwidth=rwidth, geometry=volume.geometry)
            # Add binning info to folder name if binning was applied
            prefix = 'recon'
            if custom_folder is not None:
                prefix = custom_folder
            if binning_factor > 1:
                prefix = f"{prefix}_bin{binning_factor}"

            self.save_reconstruction(volume, center_shift=center_shift,
                                     counter_offset=i * chunk_size, prefix=prefix)

    def sweep_centershift(self, center_shift_range, chunk_count=1):
        middle_slice = self.projections.shape[1] // 2
        slice_range = slice(middle_slice, middle_slice + 2)
        for center_shift in center_shift_range:
            print(f"Processing center shift {center_shift}")
            self.reconstruct(center_shift, chunk_count, custom_folder='centershift_sweep', slice_range=slice_range)

    def get_shift_filename(self, center_shift):
        offset = 1000
        adjusted_value = int((center_shift * 10) + offset)
        return f"{adjusted_value:05d}"