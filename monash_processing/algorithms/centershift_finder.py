import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
import astra


class ReconstructionCalibrator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, max_angle, pixel_size, slice_idx=None, num_projections=100):
        """
        Creates reconstructions with different center shifts and saves them as files.
        Uses 3D parallel beam geometry with FBP algorithm.

        Args:
            max_angle: Maximum angle in degrees
            pixel_size: Size of detector pixels in mm
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

        # Calculate angles (up to 180 degrees)
        all_angles = np.linspace(0, np.deg2rad(max_angle), total_projs)
        valid_angles_mask = all_angles <= np.pi
        valid_indices = np.where(valid_angles_mask)[0]

        # Select subset of indices for preview
        if len(valid_indices) > num_projections:
            indices = np.linspace(0, len(valid_indices) - 1, num_projections, dtype=int)
            valid_indices = valid_indices[indices]
            angles = all_angles[valid_indices]
        else:
            angles = all_angles[valid_angles_mask]

        # Load first projection to get dimensions
        first_proj = tifffile.imread(tiff_files[0])
        detector_rows, detector_cols = first_proj.shape

        if slice_idx is None:
            slice_idx = detector_rows // 2

        # Initialize projections array
        projections = np.zeros((len(valid_indices), detector_rows, detector_cols))

        # Load projections
        print("Loading projections...")
        for i, idx in enumerate(tqdm(valid_indices)):
            try:
                proj = tifffile.imread(tiff_files[idx])
                projections[i] = proj
            except Exception as e:
                raise RuntimeError(f"Failed to load projection {idx}: {str(e)}")

        # Create reconstructions with different shifts
        shifts = np.arange(-40, 41, 10)  # Test range from -40 to +40
        print("Computing reconstructions with different center shifts...")

        for shift in tqdm(shifts):
            # Reconstruct with center shift
            recon = self._reconstruct_slice(
                projections=projections,
                angles=angles,
                pixel_size=pixel_size,
                slice_idx=slice_idx,
                center_shift=shift,
                detector_cols=detector_cols
            )

            # Save reconstruction
            filename = preview_dir / f'center_shift_{shift:+.1f}.tiff'
            tifffile.imwrite(filename, recon.astype(np.float32))

        print(f"\nReconstructed slices saved in: {preview_dir}")
        print("Examine the files and note which shift gives the best reconstruction.")

        # Ask for user input
        while True:
            try:
                chosen_shift = float(input("\nEnter the best center shift value: "))
                break
            except ValueError:
                print("Please enter a valid number")

        return chosen_shift

    def _reconstruct_slice(self, projections, angles, pixel_size, slice_idx, center_shift, detector_cols):
        """
        Reconstruct a single slice using FBP algorithm with 3D parallel beam geometry.

        Args:
            projections: 3D numpy array of projection data (projections, rows, cols)
            angles: Array of projection angles in radians
            pixel_size: Size of detector pixels in mm
            slice_idx: Index of slice to reconstruct
            center_shift: Shift of rotation center in pixels
            detector_cols: Number of detector columns

        Returns:
            2D numpy array of reconstructed slice
        """
        # Extract the slice
        slice_projections = projections[:, slice_idx:slice_idx + 1, :]

        # Create volume geometry (single slice)
        vol_geom = astra.create_vol_geom(detector_cols, detector_cols, 1)

        # Create projection geometry with center shift
        #TODO
        center_col = detector_cols / 2 + center_shift
        proj_geom = astra.create_proj_geom('parallel',
                                           pixel_size,
                                           detector_cols,
                                           angles)

        # Create sinogram
        sino_id = astra.data2d.create('-sino', proj_geom, np.squeeze(slice_projections))

        # Create reconstruction volume
        rec_id = astra.data2d.create('-vol', vol_geom)

        proj_id = astra.create_projector('line', proj_geom, vol_geom)

        # Create FBP configuration
        cfg = astra.astra_dict('FBP')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        # Create and run the algorithm
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get the result
        result = astra.data2d.get(rec_id)

        # Clean up
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_id)

        return result  # Return the single slice