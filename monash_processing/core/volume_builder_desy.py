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

class VolumeBuilderDesy:
    def __init__(self, data_loader, original_angles, energy, prop_distance, pixel_size, is_stitched=False,
                 channel='phase',
                 detector_tilt_deg=0, show_geometry=False, sparse_factor=1, is_360_deg=False, suffix=None):
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
        self.tiff_files = None
        self.valid_angles = None
        self.valid_indices = None
        self._initialize_file_list(sparse_factor)

    def _initialize_file_list(self, sparse_factor=1):
        """Initialize the list of files and angles without loading the data."""
        if self.is_stitched:
            if self.suffix is not None:
                input_dir = self.data_loader.results_dir / (
                    f'phi_stitched_{self.suffix}' if self.channel == 'phase' else 'T_stitched')
            else:
                input_dir = self.data_loader.results_dir / ('phi_stitched' if self.channel == 'phase' else 'T_stitched')
        else:
            if self.suffix is not None:
                input_dir = self.data_loader.results_dir / (f'phi_{self.suffix}' if self.channel == 'phase' else 'T')
            else:
                input_dir = self.data_loader.results_dir / ('phi' if self.channel == 'phase' else 'T')

        if self.channel not in ['phase', 'att']:
            input_dir = self.data_loader.results_dir / self.channel
            print('Using custom input dir for channel:', self.channel)

        self.tiff_files = sorted(input_dir.glob('projection_*.tif*'))
        angles, valid_indices = self.get_valid_indices(len(self.tiff_files))

        # Apply sparse factor
        self.valid_indices = valid_indices[::sparse_factor]
        self.valid_angles = angles[::sparse_factor]

    def load_projection_slices(self, slice_range=None):
        """
        Load specific slices of projections to reduce memory usage.

        Args:
            slice_range: tuple or slice object specifying the range of slices to load
                        (e.g., (start, end) or slice(start, end))

        Returns:
            np.ndarray: Array of projections for the specified slices
            np.ndarray: Corresponding angles
        """
        if not self.tiff_files:
            raise RuntimeError("File list not initialized. Call _initialize_file_list first.")

        # Load first projection to get dimensions
        sample_data = tifffile.imread(self.tiff_files[0])
        total_height = sample_data.shape[0]

        # If no slice range specified, use full height
        if slice_range is None:
            slice_range = slice(0, total_height)
        elif isinstance(slice_range, tuple):
            slice_range = slice(*slice_range)

        projections = []
        for projection_i in tqdm(self.valid_indices, desc=f"Loading {self.channel} projections", unit="file"):
            try:
                # Load only the specified slices
                data = tifffile.imread(self.tiff_files[projection_i])[slice_range]
                projections.append(data)
            except Exception as e:
                raise RuntimeError(f"Failed to load projection from {self.tiff_files[projection_i]}: {str(e)}")

        return np.array(projections), self.valid_angles

    def reconstruct(self, center_shift=0, chunk_count=1, custom_folder=None, slice_range=None, binning_factor=1):
        """
        Reconstruct volume from projections with optional binning.

        Args:
            center_shift: Shift of rotation center
            chunk_count: Number of chunks to process
            custom_folder: Custom output folder name
            slice_range: Range of slices to process as tuple (start, end) or slice object
            binning_factor: Factor by which to bin projections (default=1, no binning)
        """
        # Load only the specified slices of projections
        projections, angles = self.load_projection_slices(slice_range)

        if binning_factor > 1:
            print(f"\nBinning projections {binning_factor}x...")
            binner = Binner(".")
            binned_projections = np.stack([
                binner._bin_2d(proj, binning_factor)
                for proj in tqdm(projections)
            ])
            projections = binned_projections
            print(f"Binned projection shape: {projections.shape}")

        if self.channel == 'att':
            projections = self.calculate_beer_lambert(projections)

        n_slices = projections.shape[1]
        chunk_size = n_slices // chunk_count
        print('Chunk size:', chunk_size)

        for i in range(chunk_count):
            print(f"Processing chunk {i + 1}/{chunk_count}")
            chunk_projections = projections[:, i * chunk_size:(i + 1) * chunk_size, :]
            print(chunk_projections.shape)

            volume = self.process_chunk(chunk_projections, angles, center_shift)
            rwidth = None

            if self.channel == 'phase':
                volume = self.convert_to_edensity(volume)
            elif self.channel == 'att':
                volume = self.convert_to_mu(volume)
                rwidth = 15

            volume = self.apply_reconstruction_ring_filter(volume, rwidth=rwidth, geometry=volume.geometry)

            prefix = 'recon'
            if custom_folder is not None:
                prefix = custom_folder
            if binning_factor > 1:
                prefix = f"{prefix}_bin{binning_factor}"

            self.save_reconstruction(volume, center_shift=center_shift,
                                     counter_offset=i * chunk_size, prefix=prefix)

    def sweep_centershift(self, center_shift_range, chunk_count=1):
        """Modified to use slice loading for memory efficiency"""
        middle_slice = tifffile.imread(self.tiff_files[0]).shape[0] // 2
        slice_range = (middle_slice, middle_slice + 2)
        for center_shift in center_shift_range:
            print(f"Processing center shift {center_shift}")
            self.reconstruct(center_shift, chunk_count, custom_folder='centershift_sweep', slice_range=slice_range)

    def get_valid_indices(self, file_count):
        print(f"File count: {file_count}")
        print(f"Original angles: {self.original_angles}")
        angle_180 = self.original_angles[file_count-1]
        print(f"Actual angle 180: {angle_180}")
        if angle_180 - 180 > 0.1:
            print(str(angle_180-180))
            raise ValueError("The 180° projection is not within 0.1 of 180 degrees!")
        return self.original_angles[:file_count], np.arange(file_count)

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
        data = self.apply_projection_ring_filter(data)
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

    def get_shift_filename(self, center_shift):
        offset = 1000
        adjusted_value = int((center_shift * 10) + offset)
        return f"{adjusted_value:05d}"
