from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, Union
import UMPA
from monash_processing.core.data_loader import DataLoader

class UMPAProcessor:
    """Wrapper for UMPA phase retrieval with projection-wise processing."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str, data_loader: DataLoader, w=1):
        self.logger = logging.getLogger(__name__)
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name
        self.data_loader = data_loader
        self.w = w

        # Output channels
        self.channels = ['dx', 'dy', 'T', 'df', 'f']

        # Create results directory
        self.results_dir = self.scan_path / 'results' / self.scan_name
        for channel in self.channels:
            (self.results_dir / channel).mkdir(parents=True, exist_ok=True)
        self.logger.info('Created channel subdirectories')

    def process_projection(self,
                           flats: np.ndarray,  # Shape: (N, X, Y)
                           projection: np.ndarray,  # Shape: (N, X, Y)
                           angle_i: int) -> Dict[str, np.ndarray]:
        """
        Process a single projection with UMPA by comparing with flat field images.

        Args:
            flats: Flat field images (N, X, Y)
            projection: Single projection data (N, X, Y)
            angle_i: Index of the current projection angle
        """
        try:
            dic = UMPA.match_unbiased(projection.astype(np.float64),
                                      flats.astype(np.float64),
                                      self.w,
                                      step=1,
                                      df=True)

            # Save results for this projection
            print('Saving results for angle', angle_i)
            for channel, data in dic.items():
                self.data_loader.save_tiff(channel, angle_i, data)

        except Exception as e:
            self.logger.error(f"UMPA processing failed for angle {angle_i}: {str(e)}")
            raise
