from typing import List, Set, Union, Optional
import h5py
from monash_processing.core.data_loader import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import re

class MultiPositionDataLoader(DataLoader):
    def __init__(self, scan_path: Union[str, Path], scan_name: str, skip_positions: Optional[Set[str]] = None):
        """
        Initialize the multi-position data loader.

        Args:
            scan_path: Path to the scan directory
            scan_name: Name of the scan
            skip_positions: Set of position indices to skip (e.g. {'03_03'})
        """
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.logger = logging.getLogger(__name__)
        self.results_dir = self.scan_path / 'results' / self.scan_name
        self.skip_positions = skip_positions or set()

        def natural_sort_key(path):
            numbers = [int(text) if text.isdigit() else text.lower()
                       for text in re.split('([0-9]+)', str(path))]
            return numbers

        # Updated pattern to match both cases:
        # 1. Files that end with .h5
        # 2. Files that end with _number.h5
        self.h5_files = sorted(
            [f for f in self.scan_path.glob(f"{scan_name}*.h5")
             if re.search(r'(\.h5$)|(_\d+\.h5$)', str(f))],
            key=natural_sort_key
        )

        if not self.h5_files:
            raise FileNotFoundError(f"No H5 files found matching pattern {scan_name}*.h5 in {scan_path}")

        self.logger.info(f"Found {len(self.h5_files)} H5 files:")
        for h5_file in self.h5_files:
            self.logger.info(f"  {h5_file}")

        # Get all available positions from first H5 file
        with h5py.File(self.h5_files[0], 'r') as f:
            scan_group = f['EXPERIMENT/SCANS']
            self.positions = sorted([pos for pos in scan_group.keys()
                                     if pos not in self.skip_positions])

        self.logger.info(f"Found positions: {self.positions}")
        if self.skip_positions:
            self.logger.info(f"Skipping positions: {self.skip_positions}")

    def _load_raw_dataset(self, h5_file: Path, dataset_path: str, position: Optional[str] = None) -> np.ndarray:
        """
        Load a specific dataset from an H5 file.

        Args:
            h5_file: Path to the H5 file
            dataset_path: Path to the dataset within the H5 file (e.g. 'FLAT_FIELD/BEFORE')
            position: Position identifier (e.g. '00_00', '01_00'). If None, uses first available position

        Returns:
            numpy array containing the dataset
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                # If no position specified, use first available position
                if position is None:
                    position = self.positions[0]

                full_path = f'EXPERIMENT/SCANS/{position}/{dataset_path}/DATA'
                data = f[full_path][:]
                return data
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_path} from {h5_file} position {position}: {str(e)}")
            raise

    def load_flat_fields(self, dark=False, position: Optional[str] = None) -> np.ndarray:
        """
        Load flat field data from all positions and combine into a single array.

        Args:
            dark: If True, load dark fields instead of flat fields
            position: Specific position to load. If None, loads all positions

        Returns:
            If dark=True: numpy array of shape (X, Y) - averaged across all positions
            If dark=False: numpy array of shape (N, X, Y) where N is number of positions
        """
        type = "flat" if not dark else "dark"
        positions_to_load = [position] if position else self.positions

        # Check if final averaged file already exists
        filename = 'averaged_flatfields.npy' if not dark else 'averaged_darkfields.npy'
        averaged_file = self.results_dir / filename

        if averaged_file.exists():
            try:
                flat_fields_array = np.load(averaged_file)
                self.logger.info(f"Loaded averaged {type} from {averaged_file}")
                return flat_fields_array
            except Exception as e:
                self.logger.error(f"Failed to load averaged {type} from {averaged_file}: {str(e)}")
                raise

        flat_fields = []

        # Load data for each position
        for pos in positions_to_load:
            for h5_file in tqdm(self.h5_files, desc=f"Loading {type} fields for position {pos}", unit="file"):
                try:
                    prefix = 'FLAT_FIELD/BEFORE' if not dark else 'DARK_FIELD/BEFORE'
                    data = self._load_raw_dataset(h5_file, prefix, pos)
                    # Average multiple exposures for this position
                    averaged_flat = self._average_fields(data)
                    flat_fields.append(averaged_flat)
                except Exception as e:
                    self.logger.error(f"Failed to load/average {type} field from {h5_file} position {pos}: {str(e)}")
                    raise

        flat_fields_array = np.array(flat_fields)  # Shape: (N, X, Y) where N is number of positions

        # For dark fields, average across all positions to get a single (X, Y) array
        if dark:
            flat_fields_array = np.mean(flat_fields_array, axis=0)  # Shape: (X, Y)
            self.logger.info(f"Averaged dark fields across positions to shape {flat_fields_array.shape}")

        # Save the final averaged array
        self._save_auxiliary_data(flat_fields_array, filename)

        self.logger.info(f"Loaded and averaged {type} fields with shape {flat_fields_array.shape}")
        return flat_fields_array

    def load_projections(self, projection_i: Optional[int] = None, step_i: Optional[int] = None,
                         position: Optional[str] = None) -> np.ndarray:
        """
        Load projection data from files for specific positions.

        Args:
            projection_i: If provided, loads only the specified projection index
            step_i: If provided, loads only from the specified file index
            position: If provided, loads only from the specified position

        Returns:
            For single projection (projection_i specified):
                - Without position specified: array of shape (N, X, Y) where N is number of positions
                - With position specified: array of shape (X, Y)
            For multiple projections:
                - Without position specified: array of shape (N, num_angles, X, Y)
                - With position specified: array of shape (num_angles, X, Y)
        """
        if step_i is not None:
            if step_i >= len(self.h5_files):
                raise ValueError(f"File index {step_i} out of range (max: {len(self.h5_files) - 1})")
            h5_files_to_load = [self.h5_files[step_i]]
        else:
            h5_files_to_load = self.h5_files

        positions_to_load = [position] if position else self.positions
        projections = []

        for h5_file in h5_files_to_load:
            file_projections = []
            try:
                with h5py.File(h5_file, 'r') as f:
                    for pos in positions_to_load:
                        dataset = f[f'EXPERIMENT/SCANS/{pos}/SAMPLE/DATA']

                        if projection_i is not None:
                            if projection_i >= dataset.shape[0]:
                                raise ValueError(f"Projection index {projection_i} out of range "
                                                 f"(max: {dataset.shape[0] - 1}) in file {h5_file}")
                            data = dataset[projection_i:projection_i + 1][0]  # Remove singleton dimension
                        else:
                            data = dataset[:]

                        file_projections.append(data)

                projections.append(file_projections)
            except Exception as e:
                self.logger.error(f"Failed to load projection from {h5_file}: {str(e)}")
                raise

        projections_array = np.array(projections)

        # Remove unnecessary dimensions
        if len(h5_files_to_load) == 1:
            projections_array = projections_array[0]  # Remove file dimension
        if position is not None:
            projections_array = projections_array[0]  # Remove position dimension when specific position requested

        self.logger.info(f"Loaded projections with shape {projections_array.shape}")
        return projections_array