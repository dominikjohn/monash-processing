import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
import astra


class ReconstructionCalibrator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, max_angle, pixel_size, slice_idx=None, num_projections=900):
        """
        Creates reconstructions with different center shifts and saves them as files.
        Uses 2D geometry with FBP algorithm and implements center shift using numpy roll.

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

        # Calculate indices to load
        indices = np.linspace(0, total_projs - 1, num_projections, dtype=int)
        angles = np.linspace(0, np.deg2rad(max_angle), total_projs)[indices]

        # Load first projection to get dimensions
        first_proj = np.array(tifffile.imread(tiff_files[0]))
        height, width = first_proj.shape

        if slice_idx is None:
            slice_idx = height // 2

        # Initialize sinogram array
        sinogram = np.zeros((len(indices), width))

        # Load projections and extract sinogram
        print("Loading projections and creating sinogram...")
        for i, idx in enumerate(tqdm(indices)):
            try:
                proj = np.array(tifffile.imread(tiff_files[idx]))
                sinogram[i, :] = proj[slice_idx, :]
            except Exception as e:
                print(f"Error loading projection {idx}: {e}")
                continue

        # Create reconstructions with different shifts
        shifts = np.arange(-40, 41, 10)  # Test range from -40 to +40
        print("Computing reconstructions with different center shifts...")

        for shift in tqdm(shifts):
            # Apply center shift using numpy roll
            shifted_sinogram = np.array([np.roll(row, int(shift)) for row in sinogram])

            # Reconstruct with shifted data
            recon = self._reconstruct_slice(shifted_sinogram, angles, pixel_size)

            # Save reconstruction
            filename = preview_dir / f'center_shift_{shift:+.1f}.tiff'
            tifffile.imwrite(filename, recon)

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

    def _reconstruct_slice(self, sinogram, angles, pixel_size):
        """
        Reconstruct a single slice using FBP algorithm with 2D geometry.

        Args:
            sinogram: 2D numpy array of projection data
            angles: Array of projection angles in radians
            pixel_size: Size of detector pixels in mm

        Returns:
            2D numpy array of reconstructed slice
        """
        n_proj, n_det = sinogram.shape

        # Create volume geometry
        vol_geom = astra.create_vol_geom(n_det, n_det)

        # Create parallel beam geometry
        proj_geom = astra.create_proj_geom('parallel', pixel_size, n_det, angles)

        # Create ASTRA objects
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        vol_id = astra.data2d.create('-vol', vol_geom)

        # Create FBP configuration
        cfg = astra.astra_dict('FBP')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id

        # Set up the FBP algorithm
        alg_id = astra.algorithm.create(cfg)

        # Run the reconstruction
        astra.algorithm.run(alg_id, 1)

        # Get the result
        result = astra.data2d.get(vol_id)

        # Clean up ASTRA objects
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(vol_id)
        astra.data2d.delete(sino_id)

        return result