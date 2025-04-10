import numpy as np
from pathlib import Path
from tqdm import tqdm
import tifffile
from typing import Union
from monash_processing.core.data_loader import DataLoader
from PIL import Image

class StitchedDataLoader(DataLoader):
    """Extends DataLoader to handle stitching of opposing projections."""

    def __init__(self, scan_path: Union[str, Path], scan_name: str, pixel_shift: int):
        super().__init__(scan_path, scan_name)
        self.pixel_shift = pixel_shift
        self.stitched_dir = self.scan_path / 'stitched' / self.scan_name

        # Create subdirectory for projections
        self.projections_dir = self.stitched_dir / 'projections'
        self.projections_dir.mkdir(parents=True, exist_ok=True)

    def load_stitched_projection(self, step_idx: int, angle_idx: int, format='tif') -> np.ndarray:
        """
        Load a single stitched projection.

        Args:
            step_idx: Index of the step
            angle_idx: Index of the projection angle

        Returns:
            2D numpy array containing the projection data
            :param format:
        """
        tiff_path = self.projections_dir / f'step_{step_idx:04d}' / f'projection_{angle_idx:04d}.{format}'

        if not tiff_path.exists():
            raise FileNotFoundError(f"Stitched projection not found: {tiff_path}")

        return tifffile.imread(tiff_path)

    def load_stitched_flat_fields(self) -> np.ndarray:
        """
        Load the stitched flat fields.

        Returns:
            Array containing the stitched flat fields
        """
        flat_path = self.stitched_dir / 'stitched_flat_fields.npy'

        if not flat_path.exists():
            raise FileNotFoundError(f"Stitched flat fields not found: {flat_path}")

        return np.load(flat_path)

    def load_stitched_dark_current(self) -> np.ndarray:
        """
        Load the stitched dark current data.

        Returns:
            Array containing the stitched dark current
        """
        dark_path = self.stitched_dir / 'stitched_dark_current.npy'

        if not dark_path.exists():
            raise FileNotFoundError(f"Stitched dark current not found: {dark_path}")

        return np.load(dark_path)

    def stitch_opposing_projections(self, steps_per_180: int) -> None:
        """
        Stitch opposing projections (180 degrees apart) with the calculated pixel shift.

        Args:
            steps_per_180: Number of projection steps for 180 degrees
        """
        self.logger.info("Starting projection stitching process...")

        # Load all necessary data
        raw_projections = self.load_projections()  # Shape: (N, num_angles, X, Y)
        flat_fields = self.load_flat_fields()  # Shape: (N, X, Y)
        dark_current = self.load_flat_fields(dark=True)  # Shape: (X, Y)

        # Process each step
        stitched_projections = []
        stitched_flats = []

        for step_idx in tqdm(range(len(self.h5_files)), desc="Stitching steps"):
            # Get projections for this step
            step_projs = raw_projections[step_idx]  # Shape: (num_angles, X, Y)
            step_flat = flat_fields[step_idx]  # Shape: (X, Y)

            # Stitch projections and flat field for this step
            stitched_step = self._stitch_step_data(
                step_projs, step_flat, dark_current, steps_per_180
            )

            stitched_projections.append(stitched_step['projections'])
            stitched_flats.append(stitched_step['flat_field'])

        # Convert to arrays
        stitched_projections = np.array(stitched_projections)
        stitched_flats = np.array(stitched_flats)

        # Save stitched data
        self._save_stitched_data(stitched_projections, stitched_flats, dark_current)

    def _stitch_step_data(self, projections: np.ndarray, flat_field: np.ndarray,
                          dark_current: np.ndarray, steps_per_180: int) -> dict:
        """
        Stitch projections and flat field for a single step.

        Args:
            projections: Projections for this step (Shape: num_angles, X, Y)
            flat_field: Flat field for this step (Shape: X, Y)
            dark_current: Dark current (Shape: X, Y)
            steps_per_180: Number of projection steps for 180 degrees

        Returns:
            Dict containing stitched projections and flat field
        """
        # Split projections into first and second half
        first_half = projections[:steps_per_180]
        second_half = projections[steps_per_180:2 * steps_per_180]

        # Process opposing projections
        stitched = []
        for i in range(steps_per_180):
            # Get opposing projections
            proj1 = first_half[i]
            proj2 = second_half[i]

            # Flip and shift the second projection
            proj2_processed = self._process_opposing_projection(proj2)

            # Combine the projections
            stitched_proj = self._combine_projections(proj1, proj2_processed)
            stitched.append(stitched_proj)

        # Stitch dark current to match dimensions
        stitched_dark = self._process_flat_field(dark_current)

        # Process flat field
        stitched_flat = self._process_flat_field(flat_field)

        return {
            'projections': np.array(stitched),
            'flat_field': stitched_flat
        }

    def _process_opposing_projection(self, projection: np.ndarray) -> np.ndarray:
        """Process opposing projection by flipping and shifting."""
        # Flip horizontally (180 degree rotation)
        flipped = np.fliplr(projection)

        # Apply vertical shift
        if self.pixel_shift > 0:
            shifted = np.pad(flipped, ((0, self.pixel_shift), (0, 0)), mode='edge')[self.pixel_shift:, :]
        else:
            shifted = np.pad(flipped, ((abs(self.pixel_shift), 0), (0, 0)), mode='edge')[:flipped.shape[0], :]

        return shifted

    def _combine_projections(self, proj1: np.ndarray, proj2: np.ndarray) -> np.ndarray:
        """
        Combine two opposing projections with weighting, only in valid overlapping regions.
        Zero or near-zero values are considered invalid data.
        """
        # Define threshold for considering a pixel as valid data
        valid_threshold = 1e-6

        # Create masks for valid data
        valid_mask1 = proj1 > valid_threshold
        valid_mask2 = proj2 > valid_threshold

        # Find overlapping region
        overlap_mask = valid_mask1 & valid_mask2

        # Initialize output array
        result = np.zeros_like(proj1)

        # Where only proj1 is valid, use proj1
        result[valid_mask1 & ~valid_mask2] = proj1[valid_mask1 & ~valid_mask2]

        # Where only proj2 is valid, use proj2
        result[~valid_mask1 & valid_mask2] = proj2[~valid_mask1 & valid_mask2]

        # In overlapping region, apply weighted average
        overlap_indices = np.where(overlap_mask)
        if len(overlap_indices[1]) > 0:  # If there is overlap
            # Find start and end of overlap region
            overlap_start = np.min(overlap_indices[1])
            overlap_end = np.max(overlap_indices[1])
            overlap_length = overlap_end - overlap_start + 1

            # Create weights for overlap region
            w1 = np.linspace(1, 0, overlap_length)
            w2 = np.linspace(0, 1, overlap_length)

            # Apply weights only in overlap region
            overlap_slice = slice(overlap_start, overlap_end + 1)
            result[:, overlap_slice] = (
                    proj1[:, overlap_slice] * w1[np.newaxis, :] +
                    proj2[:, overlap_slice] * w2[np.newaxis, :]
            )

        return result

    def _process_flat_field(self, flat_field: np.ndarray) -> np.ndarray:
        """
        Process flat field similar to projections, considering valid data regions.
        """
        # Flip and shift similar to projections
        flipped_flat = np.fliplr(flat_field)

        if self.pixel_shift > 0:
            shifted_flat = np.pad(flipped_flat, ((0, self.pixel_shift), (0, 0)), mode='edge')[self.pixel_shift:, :]
        else:
            shifted_flat = np.pad(flipped_flat, ((abs(self.pixel_shift), 0), (0, 0)), mode='edge')[
                           :flipped_flat.shape[0], :]

        # Use same combination logic as projections
        return self._combine_projections(flat_field, shifted_flat)

    def _save_stitched_data(self, projections: np.ndarray, flat_fields: np.ndarray,
                            dark_current: np.ndarray) -> None:
        """
        Save stitched data to the stitched directory.
        Projections are saved as individual TIFFs.
        Flats and darks are saved as NPY files.

        Args:
            projections: Stitched projections array (Shape: N_steps, N_angles, X, Y)
            flat_fields: Stitched flat fields array (Shape: N_steps, X, Y)
            dark_current: Dark current array (Shape: X, Y)
        """
        self.logger.info("Saving stitched data...")

        # Save projections as individual TIFF files
        for step_idx in tqdm(range(projections.shape[0]), desc="Saving projections"):
            step_dir = self.projections_dir / f'step_{step_idx:04d}'
            step_dir.mkdir(exist_ok=True)

            for angle_idx in range(projections.shape[1]):
                tif_path = step_dir / f'projection_{angle_idx:04d}.tif'
                im = Image.fromarray(projections[step_idx, angle_idx].astype(np.float32))
                im.save(tif_path)

        # Save flat fields and dark current as NPY files in stitched directory
        np.save(self.stitched_dir / 'stitched_flat_fields.npy', flat_fields)
        np.save(self.stitched_dir / 'stitched_dark_current.npy', dark_current)

        # Save metadata about the stitching process
        metadata = {
            'pixel_shift': self.pixel_shift,
            'n_steps': projections.shape[0],
            'n_angles': projections.shape[1],
            'image_shape': projections.shape[2:]
        }
        np.save(self.stitched_dir / 'stitching_metadata.npy', metadata)

        self.logger.info(f"Saved all stitched data to {self.stitched_dir}")