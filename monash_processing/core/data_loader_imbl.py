from pathlib import Path
import re
import logging
import numpy as np
from tqdm import tqdm
import h5py
from typing import Union, Optional
from monash_processing.core.data_loader import DataLoader
import re
import numpy as np
from typing import Dict

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

        # Look in the subdirectory matching scan_name
        scan_dir = self.scan_path / self.scan_name

        # Find all sample files matching the pattern SAMPLE_YI_ZJ.hdf
        self.h5_files = sorted(
            [f for f in scan_dir.glob("SAMPLE*.hdf")],
            key=natural_sort_key
        )

        if not self.h5_files:
            self.logger.warning(f"No SAMPLE files found in {self.scan_path}")

        self.logger.info(f"Found {len(self.h5_files)} HDF files:")
        for h5_file in self.h5_files:
            self.logger.info(f"  {h5_file}")

    def _get_corresponding_file(self, sample_file: Path, prefix: str) -> Path:
        """Get corresponding background or dark field file for a given sample file."""
        # Extract Y and Z values from sample filename
        # Assuming filename format like "SAMPLE_Y1_Z1.hdf"
        basename = sample_file.stem  # Gets filename without extension
        parts = basename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid sample filename format: {sample_file}")

        # Keep the Y and Z parts, replace SAMPLE with BG/DF
        new_name = f"{prefix}_{'_'.join(parts[1:])}_AFTER.hdf"
        corresponding_file = sample_file.parent / new_name

        if not corresponding_file.exists():
            new_name = f"{prefix}_{'_'.join(parts[1:])}_BEFORE.hdf"
            corresponding_file = sample_file.parent / new_name
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
                field_file = self._get_corresponding_file(h5_file, type_prefix)
                with h5py.File(field_file, 'r') as f:
                    data = f['/entry/data/data'][:]
                    if data.size > 0:
                        averaged_flat = self._average_fields(data)
                        flat_fields.append(averaged_flat)
                    else:
                        self.logger.warning(f"Empty dataset in {field_file}")
            except Exception as e:
                self.logger.error(f"Failed to load {type_prefix} field from {field_file}: {str(e)}")
                raise

        if not flat_fields:
            raise ValueError(f"No valid {type_prefix} fields found")

        flat_fields_array = np.array(flat_fields)

        if dark:  # For dark fields, average across all files
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

    def load_angles(self) -> np.ndarray:
        """Load projection angles from acquisition log file.
        Only reads angles from the first acquisition sequence.

        Returns:
            np.ndarray: Array of projection angles in degrees
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

        self.logger.info("Processed angles file not found, loading angles from acquisition log")

        # Look for acquisition log file
        log_file = self.scan_path / self.scan_name / 'acquisition.1.log'
        if not log_file.exists():
            log_file = self.scan_path / self.scan_name / 'acquisition.0.log'
            if not log_file.exists():
                raise FileNotFoundError(f"Acquisition log file not found: {log_file}")

        return IMBLDataLoader.extract_sample_angles(log_file.read_text())

    @staticmethod
    def extract_sample_angles(log_text: str) -> np.ndarray:
        """
        Extract angles from sample scans and return as 2D numpy array [n_scans, n_angles].
        Handles gaps in indices by filling a fixed-size array.
        """
        scans: Dict[str, Dict[int, float]] = {}
        current_scan: Dict[int, float] = {}
        is_sample = False
        scan_name = ""
        max_index = 0

        for line in log_text.split('\n'):
            if 'filename prefix' in line and 'SAMPLE_' in line:
                if current_scan and scan_name:
                    scans[scan_name] = current_scan
                current_scan = {}
                is_sample = True
                scan_name = re.search(r'"([^"]+)"', line).group(1)
            elif 'Acquisition finished' in line:
                if is_sample and current_scan and scan_name:
                    scans[scan_name] = current_scan
                current_scan = {}
                is_sample = False
            elif is_sample:
                # Extract both index and angle
                match = re.match(r'\d{4}-\d{2}-\d{2}.*?(\d+)\s+(\d+\.\d+)', line)
                if match:
                    idx, angle = int(match.group(1)), float(match.group(2))
                    current_scan[idx] = angle
                    max_index = max(max_index, idx)

        # Print diagnostic information
        print(f"Found {len(scans)} sample scans:")
        for name, measurements in scans.items():
            indices = list(measurements.keys())
            print(f"  {name}: {len(measurements)} measurements, indices: {min(indices)}-{max(indices)}")

        # Create array and fill values
        array_size = max_index + 1
        angle_array = np.full((len(scans), array_size), np.nan)

        for i, (name, measurements) in enumerate(sorted(scans.items())):
            for idx, angle in measurements.items():
                angle_array[i, idx] = angle

        print(f"\nFinal array shape: {angle_array.shape}")
        print("Used scans (in order):")
        for name in sorted(scans.keys()):
            print(f"  {name}")

        # Print statistics about gaps
        nan_counts = np.isnan(angle_array).sum(axis=1)
        for i, (name, nan_count) in enumerate(zip(sorted(scans.keys()), nan_counts)):
            gap_percent = (nan_count / array_size) * 100
            print(f"  {name}: {nan_count} gaps ({gap_percent:.1f}%)")

        return angle_array