import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
from monash_processing.core.volume_builder import VolumeBuilder
from monash_processing.utils.utils import Utils
from monash_processing.core.vector_reconstructor import VectorReconstructor
from monash_processing.core.chunk_manager import ChunkManager

class ReconstructionCalibrator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def find_center_shift(self, max_angle, pixel_size, slice_idx=None, num_projections=100, test_range=(-50, 50)):
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

        # Select one slice for the preview
        sliced_projections = projections[:, slice_idx:slice_idx + 1, :]

        shifts = np.linspace(test_range[0], test_range[1], 10)
        print("Computing reconstructions with different center shifts...")

        for shift in tqdm(shifts):
            print('Current shift:', shift)
            shifted_projections = Utils.apply_centershift(sliced_projections, shift)

            # Reconstruct with center shift
            recon = VolumeBuilder.reconstruct_slice(
                projections=shifted_projections,
                angles=angles,
                pixel_size=pixel_size,
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

    def find_center_shift_3d(self, max_angle, enable_short_scan, slice_idx=None, num_projections=100, test_range=(-50, 50),
                             preview_chunk_size=20):
        """
        Creates reconstructions with different center shifts using cone beam geometry and saves them as files.
        Uses 3D cone beam geometry with FDK algorithm via VectorReconstructor.

        Args:
            max_angle: Maximum angle in degrees
            pixel_size: Size of detector pixels in mm
            slice_idx: Optional specific slice to use (defaults to middle)
            num_projections: Number of projections to load for preview
            test_range: Tuple of (min, max) center shift values to test
            preview_chunk_size: Number of slices to reconstruct in each preview (middle slice will be saved)
        """
        print("Loading subset of projections for center calibration...")

        # Setup paths
        input_dir = Path(self.data_loader.results_dir) / 'phi'
        preview_dir = Path(self.data_loader.results_dir) / 'center_preview_3d'
        preview_dir.mkdir(exist_ok=True)

        # Load projections
        tiff_files = sorted(input_dir.glob('projection_*.tiff'))
        total_projs = len(tiff_files)

        # Calculate angles (up to 180 degrees)
        all_angles = np.linspace(0, np.deg2rad(max_angle), total_projs)

        # Select subset of indices for preview
        indices = np.linspace(0, len(all_angles) - 1, num_projections, dtype=int)
        angles = all_angles[indices]

        # Load first projection to get dimensions
        first_proj = tifffile.imread(tiff_files[0])
        detector_rows, detector_cols = first_proj.shape

        if slice_idx is None:
            slice_idx = detector_rows // 2

        # Calculate the chunk bounds to center around the desired slice
        chunk_start = max(0, slice_idx - preview_chunk_size // 2)
        chunk_end = min(detector_rows, chunk_start + preview_chunk_size)
        chunk_start = max(0, chunk_end - preview_chunk_size)  # Adjust start if end was capped

        # Initialize projections array
        projections = np.zeros((len(indices), detector_rows, detector_cols))

        # Load projections
        print("Loading projections...")
        for i, idx in enumerate(tqdm(indices)):
            try:
                proj = tifffile.imread(tiff_files[idx])
                projections[i] = proj
            except Exception as e:
                raise RuntimeError(f"Failed to load projection {idx}: {str(e)}")

        # Create reconstructor instance
        reconstructor = VectorReconstructor(enable_short_scan=enable_short_scan)

        shifts = np.linspace(test_range[0], test_range[1], 10)
        print("Computing reconstructions with different center shifts...")

        # Initialize chunk manager for the preview chunk
        chunk_data = projections[:, chunk_start:chunk_end, :]

        for shift in tqdm(shifts):
            print(f'Current shift: {shift}')

            # Create chunk manager with current shift
            chunk_manager = ChunkManager(
                projections=chunk_data,
                chunk_size=preview_chunk_size,
                angles=angles,
                center_shift=shift,
                channel='phase',
                debug=False,
                vector_mode=True
            )

            # Get chunk data (should be just one chunk)
            chunk_info = chunk_manager.get_chunk_data(0)

            # Reconstruct with current center shift
            reconstructor.center_shift = shift  # Update center shift
            recon = reconstructor.reconstruct_chunk(
                chunk_info['chunk_data'],
                chunk_info,
                angles,
                center_shift=shift
            )

            # Save middle slice of the reconstruction
            middle_slice_idx = preview_chunk_size // 2
            filename = preview_dir / f'center_shift_{shift:+.1f}.tiff'
            tifffile.imwrite(filename, recon[middle_slice_idx].astype(np.float32))

            # Force GPU memory cleanup
            import gc
            gc.collect()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except ImportError:
                pass

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