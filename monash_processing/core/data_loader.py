from pathlib import Path
import h5py
import numpy as np
from typing import Union, Optional
import logging
import re
import tifffile
from tqdm import tqdm


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

        if not self.h5_files:
            raise FileNotFoundError(f"No H5 files found matching pattern {scan_name}*.h5 in {scan_path}")

        self.logger.info(f"Found {len(self.h5_files)} H5 files:")
        # list the names of the h5 files
        for h5_file in self.h5_files:
            self.logger.info(f"  {h5_file}")

    def load_flat_fields(self, dark=False) -> np.ndarray:
        """Load flat field data from all files and combine, averaging multiple fields per file."""

        type = "flat" if not dark else "dark"
        filename = 'averaged_flatfields.npy' if not dark else 'averaged_darkfields.npy'

        # Check if averaged flatfield file already exists
        averaged_flatfield_file = self.results_dir / filename
        if averaged_flatfield_file.exists():
            try:
                flat_fields_array = np.load(averaged_flatfield_file)
                self.logger.info(f"Loaded averaged {type} from {averaged_flatfield_file}")
                return flat_fields_array
            except Exception as e:
                self.logger.error(f"Failed to load averaged flatfield from {averaged_flatfield_file}: {str(e)}")
                raise

        self.logger.info(f"Averaged flatfield file not found, loading and averaging flat fields from raw data")
        flat_fields = []

        for h5_file in tqdm(self.h5_files, desc=f"Loading {type} fields", unit="file"):
            try:
                prefix = 'FLAT_FIELD/BEFORE' if not dark else 'DARK_FIELD/BEFORE'
                data = self._load_raw_dataset(h5_file, prefix)
                # Average multiple flat fields for this file
                averaged_flat = self._average_fields(data)
                flat_fields.append(averaged_flat)
            except Exception as e:
                self.logger.error(f"Failed to load/average {type} field from {h5_file}: {str(e)}")
                raise

        flat_fields_array = np.array(flat_fields)  # Shape: (N, X, Y)

        # Dark fields should not change too much between steps, so we can average them
        if dark:
            flat_fields_array = np.average(flat_fields_array, axis=0)  # Shape: (X, Y)

        # Save averaged flatfields to file
        self._save_auxiliary_data(flat_fields_array, filename)

        self.logger.info(f"Loaded and averaged {type} fields with shape {flat_fields_array.shape}")

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

    def load_processed_projection(self, projection_i: int, channel: str) -> np.ndarray:
        """Load a single processed projection from a specific channel."""

        # Load from TIFF files
        tiff_path = self.results_dir / channel / f'projection_{projection_i:04d}.tiff'
        data = np.array(tifffile.imread(tiff_path))

        return data

    def save_tiff(self,
                  channel: str,
                  angle_i: int,
                  data: np.ndarray,
                  prefix='projection'):
        """Save results as separate TIFF files."""
        channel_dir = self.results_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        try:
            tiff_path = channel_dir / f'{prefix}_{angle_i:04d}.tiff'

            tifffile.imwrite(
                tiff_path,
                data
            )
        except Exception as e:
            self.logger.error(f"Failed to save tiff {angle_i} to {tiff_path}: {str(e)}")

    def perform_flatfield_correction(self, data: np.ndarray, flat_fields: np.ndarray,
                                     dark_current: np.ndarray) -> np.ndarray:
        """
        Perform flatfield correction on the data.
        """
        corrected_data = (data - dark_current) / (flat_fields - dark_current)
        return corrected_data

    def _load_raw_dataset(self, h5_file: Path, dataset_path: str) -> np.ndarray:
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
