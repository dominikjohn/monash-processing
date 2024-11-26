import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
import dask
from dask.distributed import Client, LocalCluster
from tqdm.auto import tqdm
import dask.bag as db

class ProjectionStitcher:
    """Handles stitching of processed projections with advanced intensity normalization and blending."""

    def __init__(self, data_loader, center_shift: int, blend_width: int = 10):
        """
        Initialize the EnhancedProjectionStitcher.

        Args:
            data_loader: DataLoader instance to load and save projections
            center_shift: Horizontal shift in pixels to apply between projections
            blend_width: Width of the blending region in pixels
        """
        self.data_loader = data_loader
        self.center_shift = center_shift
        self.blend_width = blend_width
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_projection(self, idx: int, channel) -> np.ndarray:
        """
        Load and prepare a projection for stitching.

        Args:
            idx: Projection index

        Returns:
            np.ndarray: Prepared projection
        """
        # Load projection using the dx channel
        proj = self.data_loader.load_processed_projection(idx, channel)
        # Apply bad pixel correction
        proj = BadPixelMask.correct_bad_pixels(proj)[0]
        return proj

    def stitch_projection_pair(self, proj_index: int, channel) -> Tuple[np.ndarray, dict]:
        """
        Stitch a pair of projections with intensity normalization and smooth blending.

        Args:
            proj_index: Index of the first projection

        Returns:
            Tuple[np.ndarray, dict]: Stitched projection and stitching statistics
        """
        # Load and prepare projections
        proj1 = self.load_and_prepare_projection(proj_index, channel)
        proj2 = self.load_and_prepare_projection(proj_index + 1800, channel)
        if channel == 'dx':
            proj2_flipped = -np.fliplr(proj2)  # Note the negative sign for phase contrast
        else:
            proj2_flipped = np.fliplr(proj2)

        # Calculate full width needed
        full_width = proj1.shape[1] + abs(self.center_shift)

        # Create aligned arrays
        p1 = np.full((proj1.shape[0], full_width), np.nan)
        p2 = np.full((proj1.shape[0], full_width), np.nan)

        # Position the projections based on the shift
        if self.center_shift >= 0:
            p2[:, :proj2_flipped.shape[1]] = proj2_flipped
            p1[:, self.center_shift:self.center_shift + proj1.shape[1]] = proj1
        else:
            p1[:, :proj1.shape[1]] = proj1
            p2[:, -self.center_shift:-self.center_shift + proj2_flipped.shape[1]] = proj2_flipped

        # Find overlap region
        overlap = ~(np.isnan(p1) | np.isnan(p2))
        overlap_cols = np.where(np.any(overlap, axis=0))[0]

        if len(overlap_cols):
            # Initialize composite
            composite = np.full_like(p1, np.nan)

            # Find overlap center and create blend region
            overlap_center = (overlap_cols[0] + overlap_cols[-1]) // 2
            blend_start = overlap_center - self.blend_width // 2
            blend_end = overlap_center + self.blend_width // 2

            # Normalize intensities in the blend region
            blend_region = (slice(None), slice(blend_start, blend_end))
            intensity_diff = np.mean(p1[blend_region]) - np.mean(p2[blend_region])
            p2 += intensity_diff  # Adjust p2 to match p1's intensity level

            # Create blend weights
            x = np.linspace(0, 1, self.blend_width)

            # Apply blending
            composite[:, :blend_start] = p2[:, :blend_start]
            composite[:, blend_end:] = p1[:, blend_end:]
            composite[blend_region] = (p1[blend_region] * x[None, :] +
                                       p2[blend_region] * (1 - x[None, :]))

            # Fill any remaining gaps
            composite = np.where(np.isnan(composite), p1, composite)
            composite = np.where(np.isnan(composite), p2, composite)

        else:
            # If no overlap, just combine the non-overlapping regions
            composite = np.where(np.isnan(p1), p2, p1)

        # Calculate statistics
        stats = {
            'shift': self.center_shift,
            'overlap_pixels': np.sum(overlap),
            'overlap_percentage': (np.sum(overlap) / proj1.size) * 100,
            'blend_width': self.blend_width,
            'total_width': full_width,
            'intensity_adjustment': intensity_diff if len(overlap_cols) else 0
        }

        return composite, stats

    def process_and_save_range(self, start_idx: int, end_idx: int, channel: str) -> list:
        """
        Process and save a range of projection pairs using Dask for parallel processing.

        Args:
            start_idx: Starting projection index
            end_idx: Ending projection index (inclusive)
            channel: Name of the input channel directory

        Returns:
            list: List of stitching statistics for each processed pair
        """
        # Set up local Dask cluster with simplified config
        cluster = LocalCluster(
            n_workers=50,
            threads_per_worker=1,
            memory_limit='4GB',  # Limit memory per worker
            lifetime_stagger='2 seconds',  # Stagger worker startups
            lifetime_restart=True,  # Automatically restart workers
        )
        client = Client(cluster)

        try:
            # Create processing function that returns None if file exists
            def process_single_projection(idx):
                # Check if output already exists
                output_path = (self.data_loader.results_dir /
                               (channel + '_stitched') /
                               f'projection_{idx:04d}.tiff')

                if output_path.exists():
                    return None

                # Process if needed
                try:
                    stitched, stats = self.stitch_projection_pair(idx, channel)
                    self.data_loader.save_tiff(
                        channel=channel + '_stitched',
                        angle_i=idx,
                        data=stitched
                    )
                    self.logger.info(f"Successfully processed {channel}-projection {idx}: {stats}")
                    return stats
                except Exception as e:
                    self.logger.error(f"Failed to process {channel} projection {idx}: {str(e)}")
                    raise

            # Create dask bag of indices and process
            indices = list(range(start_idx, end_idx + 1))
            bag = db.from_sequence(indices, npartitions=50)

            # Process and collect results
            stats_list = []
            futures = bag.map(process_single_projection).compute()

            # Filter out None results (already processed files)
            stats_list = [s for s in futures if s is not None]

            return stats_list

        finally:
            # Clean up
            client.close()
            cluster.close()