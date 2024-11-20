import numpy as np
import astra
from pathlib import Path
from tqdm import tqdm
import tifffile


class ReconstructionCalibrator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, max_angle, slice_idx=None, num_projections=900):
        """
        Creates reconstructions with different center shifts and saves them as files.

        Args:
            max_angle: Maximum angle in degrees
            slice_idx: Optional specific slice to use (defaults to middle)
            num_projections: Number of projections to load for preview
        """
        print("Loading subset of projections for center calibration...")

        # Setup paths
        input_dir = Path(self.data_loader.results_dir) / 'phi'
        preview_dir = Path(self.data_loader.results_dir) / 'center_preview'
        preview_dir.mkdir(exist_ok=True)

        # Load projections
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))
        total_projs = len(tiff_files)

        # Calculate indices to load
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

        # Extract the slice
        sinogram = projections[:, slice_idx, :]

        # Create reconstructions with different shifts
        shifts = [30, 35, 38, 40, 45]

        print("Computing reconstructions with different center shifts...")
        for shift in tqdm(shifts):
            recon = self._reconstruct_slice(sinogram, angles, shift)

            # Save reconstruction
            filename = preview_dir / f'center_shift_{shift:+.1f}.tiff'
            tifffile.imwrite(filename, recon)

        print(f"\nReconstructed slices saved in: {preview_dir}")
        print("Examine the files and note which shift gives the best reconstruction.")

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

        # Create base geometries
        vol_geom = astra.create_vol_geom(n_cols, n_cols)
        proj_geom = astra.create_proj_geom('parallel',
                                           1.0,
                                           n_cols,
                                           angles)

        proj_geom_vec = astra.geom_2vec(proj_geom)
        # Apply center of rotation correction
        proj_geom_vec = astra.geom_postalignment(proj_geom, center_shift)

        # Create ASTRA objects and reconstruct
        sino_id = astra.data2d.create('-sino', proj_geom_vec, sinogram)
        rec_id = astra.data2d.create('-vol', vol_geom)

        # Set up the reconstruction
        cfg = astra.astra_dict('SIRT')
        cfg['ProjectorId'] = astra.create_projector('line', proj_geom_vec, vol_geom)
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