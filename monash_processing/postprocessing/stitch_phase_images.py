import numpy as np
from typing import Optional, Tuple
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
from dask.distributed import Client, LocalCluster
import dask.bag as db
from scipy.ndimage import shift

class ProjectionStitcher:

    def __init__(self, data_loader, angle_spacing, center_shift):
        self.data_loader = data_loader
        self.offset = 2 * center_shift
        self.angle_spacing = angle_spacing
        self.ceil_offset = np.ceil(self.offset).astype(int)
        self.residual = self.ceil_offset - self.offset

        if center_shift < 0:
            raise ValueError("Negative center shift not supported")

    def load_and_prepare_projection(self, idx: int, channel: str) -> np.ndarray:
        proj = self.data_loader.load_processed_projection(idx, channel)
        proj = BadPixelMask.correct_bad_pixels(proj)[0]
        return proj

    def load_opposing_projections(self, idx: int, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        proj1 = self.load_and_prepare_projection(idx, channel)
        proj2 = self.load_and_prepare_projection(int(idx + self.angle_spacing * 180), channel)
        return proj1, proj2

    def normalize_projection(self, projection, part, channel):
        if part == 'right':
            mean = np.mean(projection[:, -100:-5])
        elif part == 'left':
            flip_sign = -1 if channel == 'dx' else 1
            projection = flip_sign * np.fliplr(projection)
            mean = np.mean(projection[:, 5:100])
        else:
            raise ValueError("Invalid part")

        return projection - mean

    def stitch_projection_pair(self, proj_index: int, channel: str) -> Tuple[np.ndarray, dict]:

        proj1_raw, proj2_raw = self.load_opposing_projections(proj_index, channel)
        proj1 = self.normalize_projection(proj1_raw, 'right', channel)
        proj2_flipped = self.normalize_projection(proj2_raw, 'left', channel)

        y_shape = proj1.shape[0]
        single_x_shape = proj1.shape[1]

        full_width = single_x_shape + self.ceil_offset

        p1 = np.full((y_shape, full_width), np.nan)
        p2 = np.full((y_shape, full_width), np.nan)

        p2[:, :single_x_shape] = proj2_flipped
        p1[:, self.ceil_offset:self.ceil_offset + single_x_shape] = proj1
        p1 = shift(p1, shift=[0, -self.residual], mode='reflect', order=1)

        # Find overlap region
        overlap = ~(np.isnan(p1) | np.isnan(p2))

        # Initialize composite with p1 values
        composite = p1.copy()

        # In overlap region, take the average
        composite[overlap] = (p1[overlap] + p2[overlap]) / 2

        # Fill remaining areas from p2
        composite = np.where(np.isnan(composite), p2, composite)

        return composite


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
                           f'projection_{idx:04d}.tif')

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
