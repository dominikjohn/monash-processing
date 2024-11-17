from pathlib import Path
import h5py
import numpy as np
from typing import Union, Optional
import logging
import re


class DataLoader:
    """Handles loading and organizing scan data from H5 files."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str):
        self.scan_path = Path(scan_path)
        self.logger = logging.getLogger(__name__)

        def natural_sort_key(path):
            numbers = [int(text) if text.isdigit() else text.lower()
                       for text in re.split('([0-9]+)', str(path))]
            return numbers

        self.h5_files = sorted(
            list(self.scan_path.glob(f"{scan_name}*.h5")),
            key=natural_sort_key
        )

        if not self.h5_files:
            raise FileNotFoundError(f"No H5 files found matching pattern {scan_name}*.h5 in {scan_path}")

        self.logger.info(f"Found {len(self.h5_files)} H5 files")

    def _load_dataset(self, h5_file: Path, dataset_path: str) -> np.ndarray:
        """Load a specific dataset from an H5 file."""
        try:
            with h5py.File(h5_file, 'r') as f:
                data = f['EXPERIMENT/SCANS/00_00/' + dataset_path + '/DATA'][:]
                return data
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_path} from {h5_file}: {str(e)}")
            raise

    def _average_fields(self, data: np.ndarray) -> np.ndarray:
        """Average multiple fields along the first axis."""
        if data.ndim < 3:
            raise ValueError(f"Expected at least 3D data for averaging, got shape {data.shape}")

        # Average along first axis (number of fields)
        averaged_data = np.mean(data, axis=0)
        self.logger.debug(f"Averaged fields from shape {data.shape} to {averaged_data.shape}")
        return averaged_data

    def load_dark_currents(self) -> np.ndarray:
        """Load dark field data from all files and combine, averaging multiple fields per file."""
        dark_currents = []

        for h5_file in self.h5_files:
            try:
                data = self._load_dataset(h5_file, 'DARK_FIELD/BEFORE')
                # Average multiple dark fields for this file
                averaged_dark = self._average_fields(data)
                dark_currents.append(averaged_dark)
            except Exception as e:
                self.logger.error(f"Failed to load/average dark field from {h5_file}: {str(e)}")
                raise

        dark_currents_array = np.array(dark_currents)  # Shape: (N, X, Y)
        self.logger.info(f"Loaded and averaged dark currents with shape {dark_currents_array.shape}")
        return dark_currents_array

    def load_flat_fields(self) -> np.ndarray:
        """Load flat field data from all files and combine, averaging multiple fields per file."""
        flat_fields = []

        for h5_file in self.h5_files:
            try:
                data = self._load_dataset(h5_file, 'FLAT_FIELDS/BEFORE')
                # Average multiple flat fields for this file
                averaged_flat = self._average_fields(data)
                flat_fields.append(averaged_flat)
            except Exception as e:
                self.logger.error(f"Failed to load/average flat field from {h5_file}: {str(e)}")
                raise

        flat_fields_array = np.array(flat_fields)  # Shape: (N, X, Y)
        self.logger.info(f"Loaded and averaged flat fields with shape {flat_fields_array.shape}")
        return flat_fields_array

    def load_projections(self, projection_idx: Optional[int] = None) -> np.ndarray:
        """
        Load projection data from all files.

        Args:
            projection_idx: If provided, loads only the specified projection index from each file.
                          If None, loads all projections.

        Returns:
            If projection_idx is None: 4D array with shape (N, num_angles, X, Y)
            If projection_idx is specified: 3D array with shape (N, X, Y)
            where N is the number of files
        """
        projections = []

        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    dataset = f['EXPERIMENT/SCANS/00_00/SAMPLE/DATA']

                    if projection_idx is not None:
                        # Check if projection index is valid
                        if projection_idx >= dataset.shape[0]:
                            raise ValueError(f"Projection index {projection_idx} out of range "
                                             f"(max: {dataset.shape[0] - 1}) in file {h5_file}")

                        # Load only the specified projection
                        data = dataset[projection_idx:projection_idx + 1][0]  # Remove singleton dimension
                    else:
                        # Load all projections
                        data = dataset[:]

                projections.append(data)
            except Exception as e:
                self.logger.error(f"Failed to load projection from {h5_file}: {str(e)}")
                raise

        projections_array = np.array(projections)
        self.logger.info(f"Loaded projections with shape {projections_array.shape}")
        return projections_array