from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, Union, List
import UMPA
from monash_processing.core.data_loader import DataLoader
import dask
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar

class UMPAProcessor:
    """Wrapper for UMPA phase retrieval with parallel processing using Dask."""

    def __init__(self,
                 scan_path: Union[str, Path],
                 scan_name: str,
                 data_loader: DataLoader,
                 w: float = 1,
                 n_workers: Optional[int] = None):
        """
        Initialize the UMPA processor with Dask support.

        Args:
            scan_path: Path to scan directory
            scan_name: Name of the scan
            data_loader: DataLoader instance
            w: UMPA weight parameter
            n_workers: Number of Dask workers (None for auto)
        """
        self.logger = logging.getLogger(__name__)
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.data_loader = data_loader
        self.w = w
        self.n_workers = n_workers

        # Output channels
        self.channels = ['dx', 'dy', 'T', 'df', 'f']

        # Create results directory
        self.results_dir = self.scan_path / 'results' / self.scan_name
        for channel in self.channels:
            (self.results_dir / channel).mkdir(parents=True, exist_ok=True)
        self.logger.info('Created channel subdirectories')

        # Initialize Dask client
        self.client = Client(n_workers=n_workers, threads_per_worker=1, scheduler_port=0)
        self.logger.info(f'Initialized Dask client with {len(self.client.scheduler_info()["workers"])} workers')

    def _process_single_projection(self,
                                   flats: np.ndarray,
                                   angle_i: int) -> Dict[str, np.ndarray]:
        """
        Process a single projection (to be run in parallel).
        Loads projection data within the function to optimize memory usage.
        """
        try:
            # Load the projection for this angle
            projection = self.data_loader.load_projections(projection_i=angle_i)

            dic = UMPA.match_unbiased(
                projection.astype(np.float64),
                flats.astype(np.float64),
                self.w,
                step=1,
                df=True
            )

            # Save results
            for channel, data in dic.items():
                self.data_loader.save_tiff(channel, angle_i, data)

            # Free memory explicitly
            del projection

            return {'angle': angle_i, 'status': 'success', **dic}

        except Exception as e:
            self.logger.error(f"UMPA processing failed for angle {angle_i}: {str(e)}")
            return {'angle': angle_i, 'status': 'error', 'error': str(e)}

    def process_projections(self,
                            flats: np.ndarray,
                            num_angles: int) -> List[Dict]:
        """
        Process multiple projections in parallel using Dask.
        Projections are loaded on-demand within each worker.

        Args:
            flats: Flat field images (N, X, Y)
            num_angles: Total number of angles to process

        Returns:
            List of dictionaries containing results for each projection
        """
        # Create Dask delayed objects for each angle
        delayed_results = [
            dask.delayed(self._process_single_projection)(flats, angle_i)
            for angle_i in range(num_angles)
        ]

        # Compute all results in parallel
        self.logger.info(f'Processing {num_angles} projections in parallel')
        with ProgressBar():
            results = dask.compute(*delayed_results)

        return list(results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        self.logger.info('Closed Dask client')