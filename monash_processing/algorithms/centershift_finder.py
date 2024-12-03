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

    @staticmethod
    def _get_shift_filename(shift, prefix="center_shift"):
        """
        Creates a filename for center shift reconstructions that sorts properly.
        Converts the shift value to a zero-padded string with offset to ensure proper sorting.

        Args:
            shift: The shift value
            prefix: Prefix for the filename

        Returns:
            Path object with properly formatted filename
        """
        # Use an offset to handle negative numbers (e.g., offset=1000 means -10 becomes 990)
        offset = 10000
        # Convert shift to integer (multiply by 10 to preserve one decimal place)
        adjusted_value = int((shift * 10) + offset)
        # Format with leading zeros for proper sorting
        return f"{prefix}_{adjusted_value:05d}.tif"


    def bin_projections(self, projections, binning_factor):
        """
        Bins the projections by the specified factor using average pooling.

        Args:
            projections: Input projections array (n_projections, rows, cols)
            binning_factor: Integer factor by which to bin the data

        Returns:
            Binned projections array
        """
        if binning_factor <= 1:
            return projections

        new_rows = projections.shape[1] // binning_factor
        new_cols = projections.shape[2] // binning_factor

        # Reshape and mean for binning
        binned = projections.reshape(
            projections.shape[0],  # Keep number of projections
            new_rows, binning_factor,  # Split rows
            new_cols, binning_factor  # Split columns
        ).mean(axis=(2, 4))  # Average over binning windows

        return binned

    def find_center_shift(self, max_angle, pixel_size, slice_idx=None, num_projections=100,
                          test_range=(-50, 50), stepping=10, is_stitched=True, binning_factor=1, format='tif'):
        """
        Creates reconstructions with different center shifts and saves them as files.
        Uses 3D parallel beam geometry with FBP algorithm.

        Args:
            max_angle: Maximum angle in degrees
            pixel_size: Size of detector pixels in mm
            slice_idx: Optional specific slice to use (defaults to middle)
            num_projections: Number of projections to load for preview
            binning_factor: Factor by which to bin the projections (default=1, no binning)
        """
        print("Loading subset of projections for center calibration...")

        # Setup paths
        if is_stitched:
            input_dir = Path(self.data_loader.results_dir) / 'phi_stitched'
            print('using stitched folder')
        else:
            input_dir = Path(self.data_loader.results_dir) / 'phi'

        preview_dir = Path(self.data_loader.results_dir) / 'center_preview'
        preview_dir.mkdir(exist_ok=True)

        # Load projections
        tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))
        total_projs = len(tiff_files)

        all_angles = np.linspace(0, np.deg2rad(max_angle), total_projs)
        valid_angles_mask = all_angles <= 2 * np.pi
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

        # Adjust slice_idx for binning
        if slice_idx is None:
            slice_idx = (detector_rows // binning_factor) // 2
        else:
            slice_idx = slice_idx // binning_factor

        # Initialize projections array for original size
        projections = np.zeros((len(valid_indices), detector_rows, detector_cols))

        # Load projections
        print("Loading projections...")
        for i, idx in enumerate(tqdm(valid_indices)):
            try:
                proj = tifffile.imread(tiff_files[idx])
                projections[i] = proj
            except Exception as e:
                raise RuntimeError(f"Failed to load projection {idx}: {str(e)}")

        # Apply binning if requested
        if binning_factor > 1:
            print(f"Binning projections by factor of {binning_factor}...")
            projections = self.bin_projections(projections, binning_factor)
            # Adjust pixel size for binning
            pixel_size = pixel_size * binning_factor

        # Select one slice for the preview
        sliced_projections = projections[:, slice_idx:slice_idx + 1, :]

        shifts = np.linspace(test_range[0], test_range[1], stepping)
        print("Computing reconstructions with different center shifts...")

        for shift in tqdm(shifts):
            print('Current shift:', shift)
            # Scale shift according to binning
            scaled_shift = shift / binning_factor
            shifted_projections = Utils.apply_centershift(sliced_projections, scaled_shift, order=2)

            # Reconstruct with center shift
            recon = VolumeBuilder.reconstruct_slice(
                projections=shifted_projections,
                angles=angles,
                pixel_size=pixel_size,
                is_stitched=is_stitched
            )

            # Use new filename format
            filename = preview_dir / self._get_shift_filename(shift)
            # Also save the actual shift value in a text file for reference
            with open(preview_dir / "shift_values.txt", "a") as f:
                f.write(f"File: {filename.name} -> Actual shift: {shift:.1f}\n")
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

    def find_center_shift_3d(self, max_angle, enable_short_scan, slice_idx=None, num_projections=100,
                             test_range=(-50, 50), preview_chunk_size=20, binning_factor=1, is_stitched=False, format='tif'):
        """
        Creates reconstructions with different center shifts using cone beam geometry and saves them as files.
        Uses 3D cone beam geometry with FDK algorithm via VectorReconstructor.

        Args:
            max_angle: Maximum angle in degrees
            enable_short_scan: Whether to enable short scan mode
            slice_idx: Optional specific slice to use (defaults to middle)
            num_projections: Number of projections to load for preview
            test_range: Tuple of (min, max) center shift values to test
            preview_chunk_size: Number of slices to reconstruct in each preview
            binning_factor: Factor by which to bin the projections (default=1, no binning)
        """
        print("Loading subset of projections for center calibration...")

        # Setup paths
        if is_stitched:
            input_dir = Path(self.data_loader.results_dir) / 'phi_stitched'
        else:
            input_dir = Path(self.data_loader.results_dir) / 'phi'
        preview_dir = Path(self.data_loader.results_dir) / 'center_preview_3d'
        preview_dir.mkdir(exist_ok=True)

        # Load projections
        tiff_files = sorted(input_dir.glob(f'projection_*.{format}*'))
        total_projs = len(tiff_files)

        all_angles = np.linspace(0, np.deg2rad(max_angle), total_projs)

        # Select subset of indices for preview
        indices = np.linspace(0, len(all_angles) - 1, num_projections, dtype=int)
        angles = all_angles[indices]

        print('Using angles:', angles)

        # Load first projection to get dimensions
        first_proj = tifffile.imread(tiff_files[0])
        detector_rows, detector_cols = first_proj.shape

        # Adjust slice_idx and chunk size for binning
        if slice_idx is None:
            slice_idx = (detector_rows // binning_factor) // 2
        else:
            slice_idx = slice_idx // binning_factor

        preview_chunk_size = preview_chunk_size // binning_factor

        # Calculate the chunk bounds to center around the desired slice
        chunk_start = max(0, slice_idx - preview_chunk_size // 2)
        chunk_end = min(detector_rows // binning_factor, chunk_start + preview_chunk_size)
        chunk_start = max(0, chunk_end - preview_chunk_size)

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

        # Apply binning if requested
        if binning_factor > 1:
            print(f"Binning projections by factor of {binning_factor}...")
            projections = self.bin_projections(projections, binning_factor)

        # Create reconstructor instance
        reconstructor = VectorReconstructor(enable_short_scan=enable_short_scan)

        shifts = np.linspace(test_range[0], test_range[1], 10)
        print("Computing reconstructions with different center shifts...")

        # Initialize chunk manager for the preview chunk
        chunk_data = projections[:, chunk_start:chunk_end, :]

        for shift in tqdm(shifts):
            print(f'Current shift: {shift}')

            # Scale shift according to binning
            scaled_shift = shift / binning_factor

            # Create chunk manager with current shift
            chunk_manager = ChunkManager(
                projections=chunk_data,
                chunk_size=preview_chunk_size,
                angles=angles,
                center_shift=scaled_shift,
                channel='phase',
                debug=False,
                vector_mode=True
            )

            # Get chunk data (should be just one chunk)
            chunk_info = chunk_manager.get_chunk_data(0)

            # Reconstruct with current center shift
            reconstructor.center_shift = scaled_shift  # Update center shift
            recon = reconstructor.reconstruct_chunk(
                chunk_info['chunk_data'],
                chunk_info,
                angles,
                center_shift=scaled_shift
            )

            # Save middle slice of the reconstruction
            middle_slice_idx = preview_chunk_size // 2

            filename = preview_dir / self._get_shift_filename(shift, prefix="center_shift_3d")
            # Also save the actual shift value in a text file for reference
            with open(preview_dir / "shift_values.txt", "a") as f:
                f.write(f"File: {filename.name} -> Actual shift: {shift:.1f}\n")

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