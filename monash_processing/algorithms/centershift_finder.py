# monash_processing/utils/reconstruction_utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import astra

class ReconstructionCalibrator:
    """Tools for calibrating and optimizing tomographic reconstruction parameters."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, angles, pixel_size, slice_idx=None):
        """
        Interactive tool to find center shift using a single slice.

        Args:
            angles: Array of angles in radians
            pixel_size: Pixel size in meters
            slice_idx: Optional specific slice to use (defaults to middle)

        Returns:
            float: The selected center shift value
        """
        # Load projections
        print("Loading projections for center calibration...")
        projections = self.data_loader.load_projections()[0]  # Get first step

        if slice_idx is None:
            slice_idx = projections.shape[1] // 2
            print(f"Using middle slice (index {slice_idx})")

        center_shift = self._preview_center_shift(
            projections=projections,
            angles=angles,
            pixel_size=pixel_size,
            slice_idx=slice_idx
        )

        return center_shift

    def _preview_center_shift(self, projections, angles, pixel_size, slice_idx):
        """Implementation of the interactive center shift preview."""
        # Extract the slice from all projections
        sinogram = projections[:, slice_idx, :]
        n_proj, n_cols = sinogram.shape

        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(bottom=0.25)

        def reconstruct_slice(center_shift):
            # Create geometries
            vol_geom = astra.create_vol_geom(n_cols, n_cols)
            center_col = n_cols / 2 + center_shift
            proj_geom = astra.create_proj_geom('parallel',
                                               1.0,
                                               n_cols,
                                               angles,
                                               center_col)

            # Create ASTRA objects
            sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
            rec_id = astra.data2d.create('-vol', vol_geom)

            # Set up the reconstruction
            cfg = astra.astra_dict('FBP')
            cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id

            # Run the reconstruction
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            result = astra.data2d.get(rec_id)

            # Clean up
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sino_id)

            return result

        # Initial reconstruction
        recon = reconstruct_slice(0)

        # Show sinogram and initial reconstruction
        sino_plot = ax1.imshow(sinogram, aspect='auto', cmap='gray')
        ax1.set_title('Sinogram')

        recon_plot = ax2.imshow(recon, cmap='gray')
        ax2.set_title('Reconstruction')

        # Create slider
        ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, 'Center Shift', -15, 15, valinit=0, valstep=0.5)

        def update(val):
            recon = reconstruct_slice(slider.val)
            recon_plot.set_array(recon)
            ax2.set_title(f'Reconstruction (shift: {slider.val:.1f})')
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        return slider.val