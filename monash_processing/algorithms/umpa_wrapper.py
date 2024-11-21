from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, Union, List
import UMPA
from monash_processing.core.data_loader import DataLoader
from dask import delayed, compute
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar

class UMPAProcessor:
    """Wrapper for UMPA phase retrieval with parallel processing using Dask."""

    def __init__(self,
                 scan_path: Union[str, Path],
                 scan_name: str,
                 data_loader: DataLoader,
                 w: int = 1,
                 n_workers: Optional[int] = None):
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
        self.w = w
        self.n_workers = n_workers

        # Define output channels
        self.channels = ['dx', 'dy', 'T', 'df', 'f']

        # Create results directories
        self.results_dir = self.scan_path / 'results' / self.scan_name
        for channel in self.channels:
            (self.results_dir / channel).mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Created results directory at {self.results_dir}')

    def _process_single_projection(self, angle_i: int) -> Dict[str, Union[str, int, np.ndarray]]:
        """
        Process a single projection for a specific angle.

        Args:
            flats: Flat field images.
            angle_i: Index of the projection angle.

        Returns:
            A dictionary containing the results for the angle.
        """
        try:
            # Load projection data
            projection = self.data_loader.load_projections(projection_i=angle_i)

            flats = self.data_loader.load_flat_fields()

            # Perform UMPA processing
            results = UMPA.match_unbiased(
                projection.astype(np.float64),
                flats.astype(np.float64),
                self.w,
                step=1,
                df=True
            )

            # Save results for each channel
            for channel, data in results.items():
                self.data_loader.save_tiff(channel, angle_i, data)

            return {'angle': angle_i, 'status': 'success'}

        except Exception as e:
            self.logger.error(f"Failed to process angle {angle_i}: {e}")
            return {'angle': angle_i, 'status': 'error', 'error': str(e)}

    def process_projections(self, flats: np.ndarray, num_angles: int) -> List[Dict]:
        """
        Process all projections in parallel using Dask.

        Args:
            flats: Flat field images.
            num_angles: Number of projection angles.

        Returns:
            List of dictionaries containing the results for each angle.
        """
        n_workers = self.n_workers

        cluster = LocalCluster(n_workers=n_workers)
        # No active client; create a new one
        client = Client(cluster)

        self.logger.info(f"Created a new Dask client with {self.n_workers} workers")

        # Create delayed tasks
        tasks = [
            delayed(self._process_single_projection)(angle_i)
            for angle_i in range(num_angles)
        ]

        # Compute tasks in parallel
        self.logger.info(f"Starting parallel processing of {num_angles} projections")
        with ProgressBar():
            results = compute(*tasks)

        return results