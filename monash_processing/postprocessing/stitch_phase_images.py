import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
import dask
from dask.distributed import Client, LocalCluster
from tqdm.auto import tqdm
import dask.bag as db
from scipy.interpolate import interp1d


class ProjectionStitcher:
    """Handles stitching of processed projections with edge-based normalization and average blending.
    Supports both integer and sub-pixel shifts."""

    def __init__(self, data_loader, center_shift: Union[int, float]):
        """
        Initialize the ProjectionStitcher.

        Args:
            data_loader: DataLoader instance to load and save projections
            center_shift: Horizontal shift in pixels to apply between projections (can be int or float)
        """
        self.data_loader = data_loader
        self.center_shift = center_shift
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_projection(self, idx: int, channel: str) -> np.ndarray:
        """
        Load and prepare a projection for stitching with edge-based normalization.

        Args:
            idx: Projection index
            channel: Input channel name

        Returns:
            np.ndarray: Prepared projection
        """
        # Load projection
        proj = self.data_loader.load_processed_projection(idx, channel)
        # Apply bad pixel correction
        proj = BadPixelMask.correct_bad_pixels(proj)[0]
        return proj

    def interpolate_projection(self, image: np.ndarray, output_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate a projection onto new coordinates.

        Args:
            image: Input image array
            output_coords: Target x-coordinates for each output pixel

        Returns:
            np.ndarray: Interpolated image and validity mask
        """
        input_coords = np.arange(image.shape[1])

        # Initialize output arrays
        output = np.zeros((image.shape[0], len(output_coords)))
        valid_mask = np.zeros_like(output, dtype=bool)

        # Define valid range for interpolation (slightly inside the edges)
        valid_range = (output_coords >= 0) & (output_coords <= image.shape[1] - 1)

        for i in range(image.shape[0]):
            # Create interpolation function for this row
            f = interp1d(input_coords, image[i, :], kind='cubic',
                         bounds_error=False, fill_value=np.nan)

            # Interpolate
            row_output = f(output_coords)

            # Store results
            output[i, valid_range] = row_output[valid_range]
            valid_mask[i, valid_range] = True

            # Handle edge cases by extending the edge values
            if np.any(output_coords < 0):
                output[i, output_coords < 0] = image[i, 0]
                valid_mask[i, output_coords < 0] = True
            if np.any(output_coords > image.shape[1] - 1):
                output[i, output_coords > image.shape[1] - 1] = image[i, -1]
                valid_mask[i, output_coords > image.shape[1] - 1] = True

        return output, valid_mask

    def stitch_projection_pair(self, proj_index: int, channel: str) -> Tuple[np.ndarray, dict]:
        """
        Stitch a pair of projections with edge-based normalization and average blending.
        Supports sub-pixel shifts using interpolation.

        Args:
            proj_index: Index of the first projection
            channel: Input channel name

        Returns:
            Tuple[np.ndarray, dict]: Stitched projection and stitching statistics
        """
        # Load and prepare projections
        proj1_raw = self.load_and_prepare_projection(proj_index, channel)
        proj2_raw = self.load_and_prepare_projection(proj_index + 1800, channel)

        # Normalize proj1 based on rightmost region
        right_mean1 = np.mean(proj1_raw[:, -100:-5])
        proj1 = proj1_raw - right_mean1

        # Flip and normalize proj2 based on leftmost region
        if channel == 'dx':
            proj2_flipped_raw = -np.fliplr(proj2_raw)  # Note the negative sign for phase contrast
        else:
            proj2_flipped_raw = np.fliplr(proj2_raw)

        left_mean2 = np.mean(proj2_flipped_raw[:, 5:100])
        proj2_flipped = proj2_flipped_raw - left_mean2

        # Calculate output dimensions
        total_width = proj1.shape[1] + abs(self.center_shift)
        output_coords = np.arange(int(np.ceil(total_width)))

        # Calculate coordinates for sampling each projection
        if self.center_shift >= 0:
            # For proj1, we need to sample starting at center_shift
            proj1_coords = output_coords - self.center_shift
            # For proj2_flipped, we sample from the start
            proj2_coords = output_coords
        else:
            # For proj1, we sample from the start
            proj1_coords = output_coords
            # For proj2_flipped, we need to sample starting at -center_shift
            proj2_coords = output_coords + self.center_shift

        # Interpolate both projections onto their respective coordinates
        p1, m1 = self.interpolate_projection(proj1, proj1_coords)
        p2, m2 = self.interpolate_projection(proj2_flipped, proj2_coords)

        # Find overlap region using masks
        overlap = m1 & m2

        # Create the composite image
        composite = np.zeros_like(p1)

        # In the overlap region, take the average
        composite[overlap] = (p1[overlap] + p2[overlap]) / 2

        # In non-overlap regions, take whichever image is valid
        composite[m1 & ~overlap] = p1[m1 & ~overlap]
        composite[m2 & ~overlap] = p2[m2 & ~overlap]

        # Trim any remaining empty space from the right
        valid_cols = np.where(m1.any(axis=0) | m2.any(axis=0))[0]
        if len(valid_cols) > 0:
            composite = composite[:, :valid_cols[-1] + 1]

        # Calculate statistics
        stats = {
            'shift': self.center_shift,
            'overlap_pixels': np.sum(overlap),
            'overlap_percentage': (np.sum(overlap) / proj1.size) * 100,
            'total_width': composite.shape[1],
            'right_mean1': float(right_mean1),
            'left_mean2': float(left_mean2)
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
            memory_limit='4GB',
            lifetime_stagger='2 seconds',
            lifetime_restart=True,
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
            futures = bag.map(process_single_projection).compute()

            # Filter out None results (already processed files)
            stats_list = [s for s in futures if s is not None]

            return stats_list

        finally:
            # Clean up
            client.close()
            cluster.close()