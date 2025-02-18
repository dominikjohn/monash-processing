from pathlib import Path
import h5py
import numpy as np
from typing import Union, Optional
import logging
import re
import tifffile
from tqdm import tqdm
import cv2
#from monash_processing.core.eigenflats import EigenflatManager
from PIL import Image

class DataLoader:
    """Handles loading and organizing scan data from H5 files."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str):
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.logger = logging.getLogger(__name__)
        self.results_dir = self.scan_path / 'results' / self.scan_name

        def natural_sort_key(path):
            numbers = [int(text) if text.isdigit() else text.lower()
                       for text in re.split('([0-9]+)', str(path))]
            return numbers

        # This pattern only matches scan names ending with a number
        # and therefore discards files ending with _failed.h5 or similar
        self.h5_files = sorted(
            [f for f in self.scan_path.glob(f"{scan_name}*[0-9].h5")
             if re.search(r'\d+\.h5$', str(f))],
            key=natural_sort_key
        )

        self.logger.info(f"Found {len(self.h5_files)} H5 files:")
        # list the names of the h5 files
        for h5_file in self.h5_files:
            self.logger.info(f"  {h5_file}")

    def get_save_path(self):
        return self.results_dir

    def export_raw_transmission_projections(self, min_i, max_i):
        flat_fields = self.load_flat_fields()
        dark_current = self.load_flat_fields(dark=True)
        flats_meaned = np.mean(flat_fields, axis=0)

        for i in tqdm(range(min_i, max_i)):
            meaned_proj = np.mean(self.load_projections(projection_i=i), axis=0)
            T = -np.log((meaned_proj - dark_current) / (flats_meaned - dark_current))
            self.save_tiff('T_raw', i, T)

    def load_flat_fields(self, dark=False, pca=False):
        """Load flat field data from all files and combine, averaging multiple fields per file."""

        type = "flat" if not dark else "dark"

        if pca:
            filename = 'pca_flatfields.npy'
        else:
            filename = 'averaged_flatfields.npy' if not dark else 'averaged_darkfields.npy'

        flatfield_file = self.results_dir / filename
        if flatfield_file.exists():
            try:
                flat_fields_array = np.load(flatfield_file)
                if pca:
                    mean_flats_array = np.load(self.results_dir / ('mean_' + filename))
                    self.logger.info(f"Loaded {type}field and meaned data from {flatfield_file}")
                    return flat_fields_array, mean_flats_array
                else:
                    self.logger.info(f"Loaded {type}field from {flatfield_file}")
                    return flat_fields_array
            except Exception as e:
                self.logger.error(f"Failed to load averaged flatfield from {flatfield_file}: {str(e)}")
                raise

        self.logger.info(f"Processed flatfield file not found, loading and creating flat field from raw data")
        flat_fields = []

        if dark:
            for h5_file in tqdm(self.h5_files, desc=f"Loading {type} fields", unit="file"):
                prefix = 'DARK_FIELD/BEFORE'
                data = self.load_raw_dataset(h5_file, prefix)
                averaged_flat = self._average_fields(data)
                flat_fields.append(averaged_flat)

            flat_fields_array = np.array(flat_fields)
            flat_fields_array = np.average(flat_fields_array, axis=0)
            self._save_auxiliary_data(flat_fields_array, filename)

            return flat_fields_array

        dark = self.load_flat_fields(dark=True)

        # Simple averaging of flats
        if not pca:
            for h5_file in tqdm(self.h5_files, desc=f"Loading {type} fields", unit="file"):
                prefix = 'FLAT_FIELD/BEFORE'
                data = self.load_raw_dataset(h5_file, prefix)
                data = data - dark  # Subtract dark field
                averaged_flat = self._average_fields(data)
                flat_fields.append(averaged_flat)
            flat_fields_array = np.array(flat_fields)
            self._save_auxiliary_data(flat_fields_array, filename)
            return flat_fields_array

    def load_projections(self, projection_i: Optional[int] = None, step_i: Optional[int] = None) -> np.ndarray:
        """
        Load projection data from files.

        Args:
            projection_i: If provided, loads only the specified projection index from each file.
                         If None, loads all projections.
            step_i: If provided, loads only from the specified file index in h5_files.
                    If None, loads from all files.

        Returns:
            If projection_i is None:
                - When file_i is None: 4D array with shape (N, num_angles, X, Y)
                - When file_i is specified: 3D array with shape (num_angles, X, Y)
            If projection_i is specified:
                - When file_i is None: 3D array with shape (N, X, Y)
                - When file_i is specified: 2D array with shape (X, Y)
            where N is the number of files being loaded
        """
        if step_i is not None:
            if step_i >= len(self.h5_files):
                raise ValueError(f"File index {step_i} out of range (max: {len(self.h5_files) - 1})")
            h5_files_to_load = [self.h5_files[step_i]]
        else:
            h5_files_to_load = self.h5_files

        projections = []

        for h5_file in h5_files_to_load:
            try:
                with h5py.File(h5_file, 'r') as f:
                    dataset = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA']

                    if projection_i is not None:
                        # Check if projection index is valid
                        if projection_i >= dataset.shape[0]:
                            raise ValueError(f"Projection index {projection_i} out of range "
                                             f"(max: {dataset.shape[0] - 1}) in file {h5_file}")

                        # Load only the specified projection
                        data = dataset[projection_i:projection_i + 1][0]  # Remove singleton dimension
                    else:
                        # Load all projections
                        data = dataset[:]

                projections.append(data)
            except Exception as e:
                self.logger.error(f"Failed to load projection from {h5_file}: {str(e)}")
                raise

        projections_array = np.array(projections)

        # If loading from a single file, remove the extra dimension
        if step_i is not None:
            projections_array = projections_array[0]

        self.logger.info(f"Loaded projections with shape {projections_array.shape}")
        return projections_array

    def load_angles(self) -> np.ndarray:
        """Load projection angles from all files and combine them.

        Returns:
            np.ndarray: Array of projection angles in degrees with shape (N, num_angles)
            where N is the number of files/steps
        """
        # Check if angles have already been saved
        angles_file = self.results_dir / 'projection_angles.npy'
        if angles_file.exists():
            try:
                angles = np.load(angles_file)
                self.logger.info(f"Loaded angles from {angles_file}")
                return angles
            except Exception as e:
                self.logger.error(f"Failed to load angles from {angles_file}: {str(e)}")
                raise

        self.logger.info("Processed angles file not found, loading angles from raw data")
        angles = []

        # Load angles from each file
        for h5_file in tqdm(self.h5_files, desc="Loading angles", unit="file"):
            try:
                with h5py.File(h5_file, 'r') as f:
                    file_angles = f['EXPERIMENT/SCANS/00_00/SAMPLE/ANGLES'][:]
                    angles.append(file_angles)
            except Exception as e:
                self.logger.error(f"Failed to load angles from {h5_file}: {str(e)}")
                raise

        # Stack angles from all files into 2D array
        angles_array = np.stack(angles)

        # Save the combined angles
        self._save_auxiliary_data(angles_array, 'projection_angles.npy')

        self.logger.info(f"Loaded angles array with shape {angles_array.shape}")
        return angles_array

    def load_processed_projection(self, projection_i: int, channel: str, format='tif', simple_format=False, vault_format=False) -> np.ndarray:
        """Load a single processed projection from a specific channel."""
        del simple_format  # Just for compatibility purposes
        # Load from TIFF files
        tiff_path = self.results_dir / channel / f'projection_{projection_i:04d}.{format}'
        data = np.array(tifffile.imread(tiff_path))

        return data

    def save_tiff(self,
                  channel: str,
                  angle_i: int,
                  data: np.ndarray,
                  prefix='projection',
                  subfolder=None):
        """Save results as separate TIFF files."""
        if subfolder is not None:
            channel_dir = self.results_dir / subfolder / channel
        else:
            channel_dir = self.results_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        try:
            tif_path = channel_dir / f'{prefix}_{angle_i:04d}.tif'
            im = Image.fromarray(data.astype(np.float32))
            im.save(tif_path)
        except Exception as e:
            self.logger.error(f"Failed to save tiff {angle_i} to {tif_path}: {str(e)}")

    def perform_flatfield_correction(self, data: np.ndarray, flat_fields: np.ndarray,
                                     dark_current: np.ndarray) -> np.ndarray:
        """
        Perform flatfield correction on the data.
        """
        corrected_data = (data - dark_current) / (flat_fields - dark_current)
        return corrected_data

    def load_raw_dataset(self, h5_file: Path, dataset_path: str) -> np.ndarray:
        """Load a specific dataset from an H5 file."""
        try:
            with h5py.File(h5_file, 'r') as f:
                data = f['EXPERIMENT/SCANS/00_00/' + dataset_path + '/DATA'][:]
                return data
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_path} from {h5_file}: {str(e)}")
            raise

    def _save_auxiliary_data(self, data: np.ndarray, filename: str):
        """Save auxiliary data as a separate file.

        Args:
            data: np.ndarray to save
            filename: str, name of file to save to

        Raises:
            OSError: If directory creation or file saving fails
        """
        try:
            # Ensure the results directory exists
            self.results_dir.mkdir(parents=True, exist_ok=True)

            # Create full file path and ensure it has .npy extension
            filepath = self.results_dir / (filename if filename.endswith('.npy') else f"{filename}.npy")

            # Save the data
            np.save(filepath, data)

            # Verify the file was created
            if not filepath.exists():
                raise OSError(f"File {filepath} was not created successfully")

            self.logger.info(f"Successfully saved data to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save auxiliary data to {filename}: {str(e)}")
            raise  # Re-raise the exception after logging

    def _average_fields(self, data: np.ndarray) -> np.ndarray:
        """Average multiple fields along the first axis."""
        if data.ndim < 3:
            raise ValueError(f"Expected at least 3D data for averaging, got shape {data.shape}")

        # Average along first axis (number of fields)
        averaged_data = np.mean(data, axis=0)
        self.logger.debug(f"Averaged fields from shape {data.shape} to {averaged_data.shape}")
        return averaged_data

    def load_reconstruction(self, channel, binning_factor=1) -> np.ndarray:
        # Load from TIFF files
        tiff_path = self.results_dir / channel
        if binning_factor != 1:
            tiff_path = tiff_path / f'binned{binning_factor}'
        tiff_file_list = sorted(tiff_path.glob('*.tif*'))
        data = []
        for tiff_file_name in tqdm(tiff_file_list):
            data.append(tifffile.imread(tiff_file_name))

        return np.array(data)