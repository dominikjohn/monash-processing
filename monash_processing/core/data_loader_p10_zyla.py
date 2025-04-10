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

class DataLoaderP10Zyla(DataLoader):
    """Handles loading and organizing scan data from H5 files."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str, middle_string: str = '', flat_count=25, projection_count=501, step_count=16):
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.middle_string = middle_string
        self.logger = logging.getLogger(__name__)
        self.results_dir = self.scan_path / 'processed' / self.middle_string / self.scan_name
        self.base_path = self.scan_path / 'raw' / self.middle_string / self.scan_name
        self.flat_count = flat_count
        self.projection_count = projection_count
        self.step_count = step_count

        print('Base path:', self.base_path)
        print('Results path:', self.results_dir)
        print('Flat count:', self.flat_count)

        self.master_file = glob.glob(str(self.base_path) + '/*.h5')[0]
        print('Found master file:', self.master_file)


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

    def load_projections(self, projection_i, step_i=None, proj_count=1) -> np.ndarray:
        projection = []
        for i, group in enumerate(self.filtered_groups):
            if proj_count == 1:
                projection.append(self.load_raw_dataset(group[self.flat_count + projection_i]))
            else:
                step_projections = []
                for j in range(proj_count):
                    step_projections.append(
                        self.load_raw_dataset(group[self.flat_count + projection_i * proj_count + j]))
                projection.append(np.mean(step_projections, axis=0))

        projection = np.array(projection)
        return projection

    def load_raw_dataset(self, file_path):
        import hdf5plugin
        h5_data_path = 'entry/data/data'
        f = h5py.File(file_path, 'r')
        return np.squeeze(np.array(f[h5_data_path][()], dtype=np.float64))