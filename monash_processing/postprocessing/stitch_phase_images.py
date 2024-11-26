import numpy as np
from pathlib import Path
from typing import Optional
import logging


class ProjectionStitcher:
    """Handles stitching of processed projections with specified horizontal shift."""

    def __init__(self, data_loader, center_shift: int):
        """
        Initialize the ProjectionStitcher.

        Args:
            data_loader: DataLoader instance to load and save projections
            center_shift: Horizontal shift in pixels to apply between projections
        """
        self.data_loader = data_loader
        self.center_shift = center_shift
        self.logger = logging.getLogger(__name__)

    def stitch_projection_pair(self, proj_index: int) -> np.ndarray:
        """
        Stitch a pair of projections together with the specified shift.

        Args:
            proj_index: Index of the first projection

        Returns:
            np.ndarray: Stitched projection
        """
        # Load the projection pair
        proj1 = self.data_loader.load_processed_projection(proj_index, 'phi')
        proj2 = self.data_loader.load_processed_projection(proj_index + 1800, 'phi')

        # Flip the second projection horizontally
        proj2_flipped = np.fliplr(proj2)

        # Calculate full width needed
        full_width = proj1.shape[1] + abs(self.center_shift)

        # Create empty arrays
        proj1_aligned = np.full((proj1.shape[0], full_width), np.nan)
        proj2_aligned = np.full((proj2.shape[0], full_width), np.nan)

        # Position the projections based on the shift
        if self.center_shift >= 0:
            proj2_aligned[:, :proj2_flipped.shape[1]] = proj2_flipped
            proj1_aligned[:, self.center_shift:self.center_shift + proj1.shape[1]] = proj1
        else:
            proj1_aligned[:, :proj1.shape[1]] = proj1
            proj2_aligned[:, -self.center_shift:-self.center_shift + proj2_flipped.shape[1]] = proj2_flipped

        # Create composite with clean cut
        composite = np.zeros_like(proj1_aligned)
        overlap_mask = ~(np.isnan(proj1_aligned) | np.isnan(proj2_aligned))

        # Calculate the midpoint of the overlap region
        overlap_cols = np.where(np.any(overlap_mask, axis=0))[0]
        if len(overlap_cols) > 0:
            cut_point = overlap_cols[len(overlap_cols) // 2]

            # Use proj2 for the left side of the cut point and proj1 for the right side
            composite[:, :cut_point] = np.where(
                np.isnan(proj2_aligned[:, :cut_point]),
                np.nan,
                proj2_aligned[:, :cut_point]
            )
            composite[:, cut_point:] = np.where(
                np.isnan(proj1_aligned[:, cut_point:]),
                np.nan,
                proj1_aligned[:, cut_point:]
            )
        else:
            # If no overlap, just use both projections as is
            composite[~np.isnan(proj1_aligned)] = proj1_aligned[~np.isnan(proj1_aligned)]
            composite[~np.isnan(proj2_aligned)] = proj2_aligned[~np.isnan(proj2_aligned)]

        return composite

    def process_and_save_range(self, start_idx: int, end_idx: int,
                               output_channel: str = 'stitched'):
        """
        Process and save a range of projection pairs.

        Args:
            start_idx: Starting projection index
            end_idx: Ending projection index (inclusive)
            output_channel: Name of the output channel directory
        """
        for idx in range(start_idx, end_idx + 1):
            try:
                # Stitch the projections
                stitched = self.stitch_projection_pair(idx)

                # Save the result
                self.data_loader.save_tiff(
                    channel=output_channel,
                    angle_i=idx,
                    data=stitched
                )

                self.logger.info(f"Successfully processed and saved projection {idx}")

            except Exception as e:
                self.logger.error(f"Failed to process projection {idx}: {str(e)}")
                raise