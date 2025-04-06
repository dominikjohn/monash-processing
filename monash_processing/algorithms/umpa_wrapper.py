from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, Union, List
import UMPA

import gc
import hdf5plugin
from monash_processing.core.eigenflats import EigenflatManager
from monash_processing.core.data_loader import DataLoader
from dask import delayed, compute
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar
from monash_processing.utils.utils import Utils


class UMPAProcessor:
    """Wrapper for UMPA phase retrieval with parallel processing using Dask."""

    def __init__(self,
                 scan_path: Union[str, Path],
                 scan_name: str,
                 data_loader: DataLoader,
                 w: int,
                 n_workers: Optional[int] = None,
                 slicing = None,
                 ):
        """
        Initialize the UMPA processor with Dask support.

        Args:
            scan_path: Path to the scan directory.
            scan_name: Name of the scan.
            data_loader: Instance of DataLoader for loading/saving data.
            w: UMPA weight parameter.
            n_workers: Number of Dask workers (None for automatic detection).
        """
        self.logger = logging.getLogger(__name__)
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.data_loader = data_loader
        self.slicing = slicing
        print(f"Using slicing: {self.slicing}")
        self.w = w
        print(f'Window parameter: {self.w}')
        self.n_workers = n_workers
        # Define output channels
        self.channels = ['dx', 'dy', 'T', 'df', 'f']

        # Create results directories
        self.results_dir = self.data_loader.results_dir / f'umpa_window{self.w}'
        for channel in self.channels:
            (self.results_dir / channel).mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Created results directory at {self.results_dir}')

    def _process_single_projection(self, angle_i: int, dark_future, flats_future) -> Dict[
        str, Union[str, int, np.ndarray]]:
        try:
            # Use the scattered data instead of reloading
            dark = dark_future
            flats = flats_future
            projection = (self.data_loader.load_projections(projection_i=angle_i)[self.slicing] - dark)

            print(f"Projection shape: {projection.shape}, Flats shape: {flats.shape}")

            results = UMPA.match_unbiased(
                projection.astype(np.float64),
                flats.astype(np.float64),
                self.w,
                step=1,
                df=True
            )

            # Save results for each channel
            for channel, data in results.items():
                if channel in ['T', 'dx', 'dy', 'df', 'f']:
                    self.data_loader.save_tiff(channel, angle_i, data, subfolder=f'umpa_window{self.w}')

            return {'angle': angle_i, 'status': 'success'}

        except Exception as e:
            self.logger.error(f"Failed to process angle {angle_i}: {e}")
            return {'angle': angle_i, 'status': 'error', 'error': str(e)}

    def process_projections(self, num_angles: int) -> List[Dict]:
        try:
            to_process = Utils.check_existing_files(self.results_dir, num_angles, min_size_kb=5, channel='dx')

            if not to_process:
                print("All files already processed successfully!")
                return []

            print(f"Starting parallel processing for {len(to_process)} projections...")
            print(f"Using window size: {self.w}")

            cluster = LocalCluster(n_workers=self.n_workers)
            client = Client(cluster)

            # Load shared data once
            dark = client.scatter(self.data_loader.load_flat_fields(dark=True)[self.slicing])
            flats = client.scatter(self.data_loader.load_flat_fields()[self.slicing])

            # Map each projection to a worker
            futures = client.map(
                self._process_single_projection,
                to_process,
                [dark] * len(to_process),  # Broadcast dark to each task
                [flats] * len(to_process)  # Broadcast flats to each task
            )

            # Gather results with progress bar
            with ProgressBar():
                results = client.gather(futures)

            return results

        finally:
            client.close()
            cluster.close()