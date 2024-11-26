import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask

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

    def load_and_prepare_projection(self, idx: int) -> np.ndarray:
        """
        Load and prepare a projection for stitching.

        Args:
            idx: Projection index

        Returns:
            np.ndarray: Prepared projection
        """
        # Load projection using the dx channel
        proj = self.data_loader.load_processed_projection(idx, 'dx')
        # Apply bad pixel correction
        proj = BadPixelMask.correct_bad_pixels(proj)[0]
        return proj

    def stitch_projection_pair(self, proj_index: int) -> Tuple[np.ndarray, dict]:
        """
        Stitch a pair of projections with intensity normalization and smooth blending.

        Args:
            proj_index: Index of the first projection

        Returns:
            Tuple[np.ndarray, dict]: Stitched projection and stitching statistics
        """
        # Load and prepare projections
        proj1 = self.load_and_prepare_projection(proj_index)
        proj2 = self.load_and_prepare_projection(proj_index + 1800)
        proj2_flipped = -np.fliplr(proj2)  # Note the negative sign for phase contrast

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

    def process_and_save_range(self, start_idx: int, end_idx: int,
                               output_channel: str = 'stitched') -> list:
        """
        Process and save a range of projection pairs.

        Args:
            start_idx: Starting projection index
            end_idx: Ending projection index (inclusive)
            output_channel: Name of the output channel directory
            save_stats: Whether to save stitching statistics

        Returns:
            list: List of stitching statistics for each processed pair
        """
        stats_list = []

        for idx in range(start_idx, end_idx + 1):
            try:
                # Stitch the projections
                stitched, stats = self.stitch_projection_pair(idx)
                stats_list.append(stats)

                # Save the result
                self.data_loader.save_tiff(
                    channel=output_channel,
                    angle_i=idx,
                    data=stitched
                )

                self.logger.info(f"Successfully processed projection {idx}: {stats}")

            except Exception as e:
                self.logger.error(f"Failed to process projection {idx}: {str(e)}")
                raise

        return stats_list