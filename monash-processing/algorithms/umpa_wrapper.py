from pathlib import Path
import numpy as np
import h5py
import logging
from typing import Dict, Optional, Union
import tifffile
import UMPA

class UMPAProcessor:
    """Wrapper for UMPA phase retrieval with projection-wise processing."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.scan_path = Path(scan_path)
        self.scan_name = scan_name

        # Default UMPA parameters
        self.defaults = {
            'window_size': 3,
            'save_format': 'tiff',
        }
        self.defaults.update(self.config)

        # Output channels
        self.channels = ['dx', 'dy', 'T', 'df', 'f']

        # Create results directory
        self.results_dir = self.scan_path / 'results' / self.scan_name
        for channel in self.channels:
            (self.results_dir / channel).mkdir(parents=True, exist_ok=True)

    def process_projection(self,
                           flats: np.ndarray,  # Shape: (N, X, Y)
                           projection: np.ndarray,  # Shape: (N, X, Y)
                           angle_idx: int) -> Dict[str, np.ndarray]:
        """
        Process a single projection with UMPA by comparing with flat field images.

        Args:
            flats: Flat field images (N, X, Y)
            projection: Single projection data (N, X, Y)
            angle_idx: Index of the current projection angle
        """
        try:
            # Apply UMPA
            results = self._process_single(flats, projection)

            # Save results for this projection
            self._save_results(results, angle_idx, self.results_dir)

        except Exception as e:
            self.logger.error(f"UMPA processing failed for angle {angle_idx}: {str(e)}")
            raise

    def _process_single(self,
                        reference: np.ndarray,
                        sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a single projection with UMPA."""

        dic = UMPA.match_unbiased(sample.astype(np.float64),
                                  reference.astype(np.float64),
                                  3,
                                  step=1,
                                  df=True)

        # Placeholder for actual UMPA processing
        results = {
            'dx': None,  # Differential phase x
            'dy': None,  # Differential phase y
            'T': None,  # Transmission signal
            'df': None,  # Dark-field signal
            'f': None  # Quality metrics
        }

        return results

    def _save_results(self,
                      results: Dict[str, np.ndarray],
                      angle_idx: int,
                      save_dir: Path):
        """Save results for one projection."""
        if self.defaults['save_format'] == 'h5':
            self._save_as_h5(results, angle_idx, save_dir)
        else:
            self._save_as_tiff(results, angle_idx, save_dir)

    def _save_as_h5(self,
                    results: Dict[str, np.ndarray],
                    angle_idx: int,
                    save_dir: Path):
        """Save results as H5 file."""
        h5_path = save_dir / f'projection_{angle_idx:04d}.h5'
        with h5py.File(h5_path, 'w') as f:
            for channel, data in results.items():
                f.create_dataset(
                    channel,
                    data=data,
                    compression='gzip',
                    compression_opts=4
                )

    def _save_as_tiff(self,
                      results: Dict[str, np.ndarray],
                      angle_idx: int,
                      save_dir: Path):
        """Save results as separate TIFF files."""
        for channel, data in results.items():
            tiff_path = save_dir / channel / f'projection_{angle_idx:04d}.tiff'
            tifffile.imwrite(
                tiff_path,
                data,
                compression='zlib',
                compressionlevel=8
            )

    def load_results(self,
                     results_dir: Path,
                     num_angles: int) -> Dict[str, np.ndarray]:
        """
        Load processed results back into memory.
        Returns 3D arrays (num_angles, X, Y) for each channel.
        """
        results = {}
        results_dir = Path(results_dir)

        if self.defaults['save_format'] == 'h5':
            # Load from H5 files
            for angle in range(num_angles):
                h5_path = results_dir / f'projection_{angle:04d}.h5'
                with h5py.File(h5_path, 'r') as f:
                    for channel in self.channels:
                        if channel not in results:
                            results[channel] = []
                        results[channel].append(f[channel][:])
        else:
            # Load from TIFF files
            for channel in self.channels:
                channel_data = []
                for angle in range(num_angles):
                    tiff_path = results_dir / channel / f'projection_{angle:04d}.tiff'
                    channel_data.append(tifffile.imread(tiff_path))
                results[channel] = np.array(channel_data)

        # Convert lists to arrays
        results = {k: np.array(v) for k, v in results.items()}
        return results