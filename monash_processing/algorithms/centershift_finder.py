import numpy as np
import matplotlib.pyplot as plt
import astra
from pathlib import Path
from tqdm import tqdm
import tifffile


class ReconstructionCalibrator:
    """Tools for calibrating and optimizing tomographic reconstruction parameters."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, max_angle, pixel_size, slice_idx=None, num_projections=900):
        """
        Shows reconstructions with different center shifts in a grid.

        Args:
            max_angle: Maximum angle in degrees
            pixel_size: Pixel size in meters
            slice_idx: Optional specific slice to use (defaults to middle)
            num_projections: Number of projections to load for preview
        """
        print("Loading subset of projections for center calibration...")

        # Properly handle path
        input_dir = Path(self.data_loader.results_dir) / 'phi'
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))
        total_projs = len(tiff_files)

        # Calculate indices to load (evenly spaced)
        indices = np.linspace(0, total_projs - 1, num_projections, dtype=int)
        angles = np.linspace(0, np.deg2rad(max_angle), total_projs)[indices]

        projections = []
        for idx in tqdm(indices, desc="Loading projections"):
            try:
                data = np.array(tifffile.imread(tiff_files[idx]))
                projections.append(data)
            except Exception as e:
                print(f"Error loading projection {idx}: {e}")
                continue

        projections = np.array(projections)

        if slice_idx is None:
            slice_idx = projections.shape[1] // 2

        # Extract the slice from all projections
        sinogram = projections[:, slice_idx, :]
        n_proj, n_cols = sinogram.shape

        # Create a grid of reconstructions with different shifts
        shifts = [-10, -5, -2, 0, 2, 5, 10]
        reconstructions = []

        print("Computing reconstructions with different center shifts...")
        for shift in tqdm(shifts):
            reconstructions.append(self._reconstruct_slice(sinogram, angles, shift))

        # Display results
        fig, axes = plt.subplots(1, len(shifts), figsize=(20, 4))
        fig.suptitle('Center Shift Preview - Close window and enter chosen shift value')

        for ax, recon, shift in zip(axes, reconstructions, shifts):
            ax.imshow(recon, cmap='gray')
            ax.set_title(f'Shift: {shift}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Ask for user input
        while True:
            try:
                chosen_shift = float(input("\nEnter the best center shift value (-10 to 10): "))
                if -10 <= chosen_shift <= 10:
                    break
                print("Shift must be between -10 and 10")
            except ValueError:
                print("Please enter a valid number")

        return chosen_shift

    def _reconstruct_slice(self, sinogram, angles, center_shift):
        """Reconstruct a single slice with given center shift."""
        n_proj, n_cols = sinogram.shape

        # Create geometries
        vol_geom = astra.create_vol_geom(n_cols, n_cols)
        center_col = n_cols / 2 + center_shift
        proj_geom = astra.create_proj_geom('parallel',
                                           self.pixel_size,
                                           n_cols,
                                           angles)

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