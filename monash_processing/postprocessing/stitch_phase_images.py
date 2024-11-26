import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
        Uses averaging in the overlap region.

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

        # Create composite with averaging in overlap region
        composite = np.zeros_like(proj1_aligned)

        # Create masks for valid (non-nan) regions
        mask1 = ~np.isnan(proj1_aligned)
        mask2 = ~np.isnan(proj2_aligned)
        overlap_mask = mask1 & mask2

        # Fill non-overlapping regions
        composite[mask1 & ~mask2] = proj1_aligned[mask1 & ~mask2]
        composite[mask2 & ~mask1] = proj2_aligned[mask2 & ~mask1]

        # Average the overlapping regions
        composite[overlap_mask] = (proj1_aligned[overlap_mask] + proj2_aligned[overlap_mask]) / 2.0

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

    def visualize_alignment(self, proj_index: int, shift_range: Tuple[int, int] = (400, 2000)) -> Tuple[
        plt.Figure, Slider]:
        """
        Create an interactive visualization of projection alignment with a slider to control shift.

        Args:
            proj_index: Index of the projection to visualize
            shift_range: Tuple of (min_shift, max_shift) for the slider

        Returns:
            Tuple of (matplotlib figure, slider widget)
        """
        # Load projections
        proj1 = self.data_loader.load_processed_projection(proj_index, 'phi')
        proj2 = self.data_loader.load_processed_projection(proj_index + 1800, 'phi')
        proj2_flipped = np.fliplr(proj2)

        # Create figure and subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2, height_ratios=[0.1, 1, 1])

        # Create slider axis
        slider_ax = fig.add_subplot(gs[0, :])

        # Create image axes
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])

        # Initialize with current shift value
        initial_shift = self.center_shift

        # Create initial static plots
        ax1.imshow(proj1, cmap='gray')
        ax1.set_title('Projection 1')

        ax2.imshow(proj2_flipped, cmap='gray')
        ax2.set_title('Projection 2 (Flipped)')

        def update_alignment(shift):
            """Update the visualization with new shift value"""
            # Clear previous plots
            ax3.clear()
            ax4.clear()

            # Calculate full width needed
            full_width = proj1.shape[1] + abs(int(shift))

            # Create empty arrays
            proj1_aligned = np.full((proj1.shape[0], full_width), np.nan)
            proj2_aligned = np.full((proj2.shape[0], full_width), np.nan)

            # Position the projections
            if shift >= 0:
                proj2_aligned[:, :proj2_flipped.shape[1]] = proj2_flipped
                proj1_aligned[:, int(shift):int(shift) + proj1.shape[1]] = proj1
            else:
                proj1_aligned[:, :proj1.shape[1]] = proj1
                proj2_aligned[:, -int(shift):-int(shift) + proj2_flipped.shape[1]] = proj2_flipped

            # Create composite with averaging in overlap region
            composite = np.zeros_like(proj1_aligned)

            # Create masks for valid (non-nan) regions
            mask1 = ~np.isnan(proj1_aligned)
            mask2 = ~np.isnan(proj2_aligned)
            overlap_mask = mask1 & mask2

            # Fill non-overlapping regions
            composite[mask1 & ~mask2] = proj1_aligned[mask1 & ~mask2]
            composite[mask2 & ~mask1] = proj2_aligned[mask2 & ~mask1]

            # Average the overlapping regions
            composite[overlap_mask] = (proj1_aligned[overlap_mask] + proj2_aligned[overlap_mask]) / 2.0

            # Update plots
            ax3.imshow(proj1_aligned, cmap='gray')
            ax3.set_title('Projection 1 (Aligned)')

            ax4.imshow(composite, cmap='gray')
            ax4.set_title('Composite (Averaged Overlap)')

            # Show overlap region boundaries
            if np.any(overlap_mask):
                overlap_cols = np.where(np.any(overlap_mask, axis=0))[0]
                if len(overlap_cols) > 0:
                    ax4.axvline(x=overlap_cols[0], color='r', linestyle='--', alpha=0.5, label='Overlap start')
                    ax4.axvline(x=overlap_cols[-1], color='r', linestyle='--', alpha=0.5, label='Overlap end')
                    ax4.legend()

            # Update overlap information
            overlap_pixels = np.sum(overlap_mask)
            overlap_percentage = (overlap_pixels / (proj1.shape[0] * proj1.shape[1])) * 100

            # Update text
            if hasattr(fig, 'text_artist'):
                fig.text_artist.remove()
            fig.text_artist = fig.text(0.02, 0.02,
                                       f'Shift: {int(shift)} pixels\n'
                                       f'Overlap: {overlap_pixels} pixels ({overlap_percentage:.1f}%)\n'
                                       f'Overlap width: {len(overlap_cols) if np.any(overlap_mask) else 0} pixels\n'
                                       f'Total width: {full_width} pixels',
                                       bbox=dict(facecolor='white', alpha=0.8))

            fig.canvas.draw_idle()

        # Create slider
        slider = Slider(slider_ax, 'Horizontal Shift',
                        shift_range[0], shift_range[1],
                        valinit=initial_shift,
                        valstep=1)
        slider.on_changed(update_alignment)

        # Initial update
        update_alignment(initial_shift)

        plt.tight_layout()
        plt.show()

        return fig, slider