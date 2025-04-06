from pathlib import Path
import h5py
import numpy as np
from typing import Union
import logging
import tifffile
from tqdm import tqdm
import glob

from monash_processing.core.data_loader import DataLoader
from PIL import Image
from collections import defaultdict
import re

class DataLoaderP10(DataLoader):
    """Handles loading and organizing scan data from H5 files."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str, middle_string: str = '', flat_count=25, projection_count=501):
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.middle_string = middle_string
        self.logger = logging.getLogger(__name__)
        self.results_dir = self.scan_path / 'processed' / self.middle_string / self.scan_name
        self.base_path = self.scan_path / 'raw' / self.middle_string / self.scan_name
        self.flat_count = flat_count
        self.projection_count = projection_count

        print('Base path:', self.base_path)
        print('Results path:', self.results_dir)
        print('Flat count:', self.flat_count)

        self.filtered_groups = []

        file_list = sorted(glob.glob(str(self.base_path) + '/*_data_*.h5'))
        # Group files based on the pattern before "_data_"
        grouped_files = defaultdict(list)
        for file_path in file_list:
            match = re.search(r'_(\d{5})_data_', file_path)
            if match:
                group_id = match.group(1)  # finds the 0000X part
                grouped_files[group_id].append(file_path)

        i = 0
        for group_id, files in sorted(grouped_files.items()):
            if len(files) >= 200:
                print(group_id)
                self.filtered_groups.append(files)
                i += 1

    def get_save_path(self):
        return self.results_dir

    def load_flat_fields(self, dark=False, pca=False):

        filename = 'averaged_flatfields.npy' if not dark else 'averaged_darkfields.npy'

        flatfield_file = self.results_dir / filename
        if flatfield_file.exists():
            try:
                print('Flatfield file exists:', flatfield_file)
                flat_fields_array = np.load(flatfield_file)
                self.logger.info(f"Loaded from {flatfield_file}")
                return flat_fields_array
            except Exception as e:
                self.logger.error(f"Failed to load averaged flatfield from {flatfield_file}: {str(e)}")
                raise

        self.logger.info(f"Processed flatfield file not found, loading and creating flat field from raw data")

        all_flats = []

        for i, group in enumerate(self.filtered_groups):
            if dark:
                flats_shape = self.load_raw_dataset(group[0]).shape
                dark = np.zeros(flats_shape)
                print('Created fake dark field filled with zeros:', dark.shape)
                self._save_auxiliary_data(dark, 'averaged_darkfields.npy')
                return dark
            print('Processing group:', i)
            flats_before_list = group[:self.flat_count]
            flats_after_list = group[self.flat_count+self.projection_count:self.flat_count*2+self.projection_count]
            flats_before = [self.load_raw_dataset(file) for file in tqdm(flats_before_list)]
            print('Flats before:', len(flats_before_list))
            flats_after = [self.load_raw_dataset(file) for file in tqdm(flats_after_list)]
            print('Flats after:', len(flats_before_list))
            flats = np.mean(np.array(flats_before + flats_after), axis=0)
            all_flats.append(flats)

        all_flats = np.array(all_flats)

        self._save_auxiliary_data(all_flats, 'averaged_flatfields.npy')

        return all_flats

    def load_raw_dataset(self, file_path):
        h5_data_path = 'entry/data/data'
        f = h5py.File(file_path, 'r')
        return np.squeeze(np.array(f[h5_data_path][()], dtype=np.float64))

    def load_projections(self, projection_i, step_i) -> np.ndarray:
        projection = []
        for i, group in enumerate(self.filtered_groups):
            projection.append(self.load_raw_dataset(group[self.flat_count + projection_i]))

        projection = np.array(projection)
        return projection

    def load_processed_projection(self, projection_i: int, channel: str, format='tif', simple_format=False) -> np.ndarray:
        """Load a single processed projection from a specific channel."""
        # Load from TIFF files
        if simple_format:
            tiff_path = self.results_dir / channel / f'projection_{projection_i:04d}.{format}'
        else:
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