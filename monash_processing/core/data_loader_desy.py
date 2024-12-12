from pathlib import Path
import h5py
import numpy as np
from typing import Union
import logging
import re
import tifffile
from tqdm import tqdm

from monash_processing.core.data_loader import DataLoader
from PIL import Image

class DataLoaderDesy(DataLoader):
    """Handles loading and organizing scan data from H5 files."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str):
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.logger = logging.getLogger(__name__)
        self.results_dir = self.scan_path / self.scan_name

    def get_save_path(self):
        return self.results_dir

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

    def load_processed_projection(self, projection_i: int, channel: str, format='tif') -> np.ndarray:
        """Load a single processed projection from a specific channel."""

        # Load from TIFF files
        tiff_path = self.results_dir / channel / f'{channel}_{projection_i:04d}.{format}'
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
            tif_path = channel_dir / f'{prefix}_{angle_i:04d}.tif'
            im = Image.fromarray(data.astype(np.float32))
            im.save(tif_path)
        except Exception as e:
            self.logger.error(f"Failed to save tiff {angle_i} to {tif_path}: {str(e)}")
