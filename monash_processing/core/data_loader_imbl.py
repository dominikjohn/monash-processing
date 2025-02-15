from pathlib import Path
import re
import logging
import numpy as np
from tqdm import tqdm
import h5py
from typing import Union, Optional
from monash_processing.core.data_loader import DataLoader

class IMBLDataLoader(DataLoader):
    """Handles loading and organizing scan data from HDF files for IMBL beamline."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str):
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.logger = logging.getLogger(__name__)

        # Extract day from input path and construct output path
        day_match = re.search(r'Day\d+', str(scan_path))
        if not day_match:
            raise ValueError("Input path must contain 'DayX' pattern")
        day = day_match.group()

        # Construct output path
        input_parts = str(scan_path).split('input')
        if len(input_parts) != 2:
            raise ValueError("Input path must contain 'input' directory")

        self.results_dir = Path(input_parts[0]) / 'output' / day / scan_name

        def natural_sort_key(path):
            numbers = [int(text) if text.isdigit() else text.lower()
                       for text in re.split('([0-9]+)', str(path))]
            return numbers

        # Find all sample files matching the pattern SAMPLE_YI_ZJ.hdf
        sample_pattern = f"{scan_name}_Y*_Z*.hdf"
        self.h5_files = sorted(
            [f for f in self.scan_path.glob(sample_pattern)
             if re.search(r'Y\d+_Z\d+\.hdf$', str(f))],
            key=natural_sort_key
        )

        self.logger.info(f"Found {len(self.h5_files)} HDF files:")
        for h5_file in self.h5_files:
            self.logger.info(f"  {h5_file}")

    def _get_corresponding_file(self, sample_file: Path, prefix: str) -> Path:
        """Get corresponding background or dark field file for a given sample file."""
        # Extract Y and Z values from sample filename
        match = re.search(r'Y(\d+)_Z(\d+)\.hdf$', str(sample_file))
        if not match:
            raise ValueError(f"Invalid sample filename pattern: {sample_file}")

        y_val, z_val = match.groups()
        corresponding_file = sample_file.parent / f"{prefix}_Y{y_val}_Z{z_val}_AFTER.hdf"

        if not corresponding_file.exists():
            raise FileNotFoundError(f"Corresponding {prefix} file not found: {corresponding_file}")

        return corresponding_file

    def load_flat_fields(self, dark=False, pca=False):
        """Load flat field data from all files."""
        if pca:
            raise NotImplementedError("PCA flatfield option is not supported in IMBLDataLoader")

        type_prefix = "DF" if dark else "BG"
        filename = f'averaged_{"dark" if dark else "flat"}fields.npy'

        flatfield_file = self.results_dir / filename
        if flatfield_file.exists():
            try:
                flat_fields_array = np.load(flatfield_file)
                self.logger.info(f"Loaded {type_prefix} field from {flatfield_file}")
                return flat_fields_array
            except Exception as e:
                self.logger.error(f"Failed to load averaged {type_prefix} field from {flatfield_file}: {str(e)}")
                raise

        self.logger.info(f"Processed {type_prefix} field file not found, loading from raw data")
        flat_fields = []

        for h5_file in tqdm(self.h5_files, desc=f"Loading {type_prefix} fields", unit="file"):
            try:
                # Get corresponding background/dark field file
                field_file = self._get_corresponding_file(h5_file, type_prefix)

                # Load and process the data
                with h5py.File(field_file, 'r') as f:
                    data = f['/entry/data/data'][:]  # Image data path
                    averaged_flat = self._average_fields(data)
                    flat_fields.append(averaged_flat)

            except Exception as e:
                self.logger.error(f"Failed to load {type_prefix} field from {field_file}: {str(e)}")
                raise

        flat_fields_array = np.array(flat_fields)
        if not dark:  # For flat fields, average across all files
            flat_fields_array = np.average(flat_fields_array, axis=0)

        self._save_auxiliary_data(flat_fields_array, filename)
        return flat_fields_array

    def load_projections(self, projection_i: Optional[int] = None, step_i: Optional[int] = None) -> np.ndarray:
        """Load projection data from HDF files."""
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
                    dataset = f['/entry/data/data']  # Image data path

                    if projection_i is not None:
                        if projection_i >= dataset.shape[0]:
                            raise ValueError(f"Projection index {projection_i} out of range "
                                             f"(max: {dataset.shape[0] - 1}) in file {h5_file}")
                        data = dataset[projection_i:projection_i + 1][0]
                    else:
                        data = dataset[:]

                projections.append(data)
            except Exception as e:
                self.logger.error(f"Failed to load projection from {h5_file}: {str(e)}")
                raise

        projections_array = np.array(projections)

        if step_i is not None:
            projections_array = projections_array[0]

        self.logger.info(f"Loaded projections with shape {projections_array.shape}")
        return projections_array