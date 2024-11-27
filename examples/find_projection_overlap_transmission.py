from monash_processing.core.multi_position_data_loader import MultiPositionDataLoader
import numpy as np
from scipy.signal import correlate
from typing import Tuple, NamedTuple
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


class ProjectionPair(NamedTuple):
    proj1: np.ndarray
    proj2: np.ndarray
    corrected1: np.ndarray
    corrected2: np.ndarray

class ProjectionAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def load_and_correct_projections(self, angle_index1: int, angle_index2: int) -> ProjectionPair:
        """Load and perform flatfield correction on a pair of projections."""
        # Load the projections
        proj1 = self.data_loader.load_projections(angle_index1)
        proj2 = self.data_loader.load_projections(angle_index2)

        # Load correction data
        flatfields = self.data_loader.load_flat_fields()
        dark_current = self.data_loader.load_flat_fields(dark=True)

        # Perform corrections
        proj1_cor = self.data_loader.perform_flatfield_correction(proj1, flatfields, dark_current)
        proj2_cor = self.data_loader.perform_flatfield_correction(proj2, flatfields, dark_current)

        return ProjectionPair(proj1, proj2, proj1_cor, proj2_cor)

    def find_projection_offset(self, angle_index1: int, angle_index2: int) -> Tuple[int, float]:
        """Find the horizontal shift between two projections using cross-correlation."""
        print('Loading and correcting projections...')
        pair = self.load_and_correct_projections(angle_index1, angle_index2)

        # Take mean along first axis if we have multiple images
        proj1_cor = np.mean(pair.corrected1, axis=0)
        proj2_cor = np.mean(pair.corrected2, axis=0)

        # Flip the second projection horizontally
        proj2_flipped = np.fliplr(proj2_cor)

        # Calculate center region to analyze
        center_y = proj1_cor.shape[0] // 2
        region_height = proj1_cor.shape[0] // 2  # Use half the image height
        start_y = center_y - region_height // 2
        end_y = center_y + region_height // 2

        # Use more rows for better statistics
        num_rows = min(50, region_height)
        rows_to_check = np.linspace(start_y, end_y, num_rows, dtype=int)

        shifts = []
        correlations = []

        # Correlate each pair of rows
        for y in tqdm(rows_to_check, desc='Calculating correlations'):
            row1 = proj1_cor[y, :]
            row2 = -proj2_flipped[y, :]  # Negative sign for phase contrast

            # Normalize rows to zero mean and unit variance
            row1 = (row1 - np.mean(row1)) / (np.std(row1) + 1e-10)
            row2 = (row2 - np.mean(row2)) / (np.std(row2) + 1e-10)

            correlation = correlate(row1, row2, mode='full')
            shift = correlation.argmax() - (len(row1) - 1)

            shifts.append(shift)
            correlations.append(correlation.max())

        shifts = np.array(shifts)
        correlations = np.array(correlations)

        # Filter out outliers using median absolute deviation
        median_shift = np.median(shifts)
        mad = np.median(np.abs(shifts - median_shift))
        mask = np.abs(shifts - median_shift) < 3 * mad  # 3 sigma rule

        # Use filtered data if we have any points left, otherwise use all data
        if np.sum(mask) > 0:
            optimal_shift = int(np.median(shifts[mask]))
            max_correlation = np.mean(correlations[mask])
        else:
            optimal_shift = int(median_shift)
            max_correlation = np.mean(correlations)

        # Print statistics for debugging
        print(f"\nShift statistics:")
        print(f"Raw shifts: mean={np.mean(shifts):.1f}, median={np.median(shifts):.1f}, "
              f"std={np.std(shifts):.1f}")
        print(f"Number of rows analyzed: {len(shifts)}")
        print(f"Number of shifts after outlier filtering: {np.sum(mask)}")
        print(f"Correlation values: min={np.min(correlations):.3f}, "
              f"max={np.max(correlations):.3f}, mean={np.mean(correlations):.3f}")

        return optimal_shift, max_correlation

    def visualize_alignment(self, angle_index1: int, angle_index2: int, shift: int):
        """Visualize the alignment of phase contrast projections."""
        pair = self.load_and_correct_projections(angle_index1, angle_index2)
        proj1_cor = np.mean(pair.corrected1, axis=0)
        proj2_cor = np.mean(pair.corrected2, axis=0)

        proj2_flipped = np.fliplr(proj2_cor)

        # Calculate full width needed for shifted images
        full_width = proj1_cor.shape[1] + abs(shift)

        # Create empty arrays for aligned projections
        proj1_aligned = np.full((proj1_cor.shape[0], full_width), np.nan)
        proj2_aligned = np.full((proj2_cor.shape[0], full_width), np.nan)

        # Position the projections
        if shift >= 0:
            proj2_aligned[:, :proj2_flipped.shape[1]] = proj2_flipped
            proj1_aligned[:, shift:shift + proj1_cor.shape[1]] = proj1_cor
        else:
            proj1_aligned[:, :proj1_cor.shape[1]] = proj1_cor
            proj2_aligned[:, abs(shift):abs(shift) + proj2_flipped.shape[1]] = proj2_flipped

        # Create composite image
        composite = np.zeros_like(proj1_aligned)
        overlap_mask = ~(np.isnan(proj1_aligned) | np.isnan(proj2_aligned))
        composite[~np.isnan(proj1_aligned)] = proj1_aligned[~np.isnan(proj1_aligned)]
        composite[~np.isnan(proj2_aligned)] = proj2_aligned[~np.isnan(proj2_aligned)]
        composite[overlap_mask] = (proj1_aligned[overlap_mask] + proj2_aligned[overlap_mask]) / 2

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Projection Alignment\nAngles: {angle_index1} and {angle_index2}, Shift: {shift} pixels')

        vmin, vmax = proj1_cor.min(), proj1_cor.max()
        for ax, data, title in [
            (axes[0, 0], proj1_cor, 'Projection 1'),
            (axes[0, 1], proj2_flipped, 'Projection 2 (Flipped)'),
            (axes[1, 0], proj1_aligned, 'Projection 1 (Aligned)'),
            (axes[1, 1], composite, 'Composite')
        ]:
            im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        # Add overlap information
        overlap_pixels = np.sum(overlap_mask)
        overlap_percentage = (overlap_pixels / (proj1_cor.shape[0] * proj1_cor.shape[1])) * 100
        fig.text(0.02, 0.02,
                 f'Overlap: {overlap_pixels} pixels ({overlap_percentage:.1f}%)\n'
                 f'Total width: {full_width} pixels',
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig


# Example usage:
scan_path = Path("/data/mct/22203/")
scan_name = "K21N_sample"

# Calculate indices for opposing projections
total_angles = 3640
angle_per_step = 364 / total_angles
steps_per_180 = int(180 / angle_per_step)

loader = MultiPositionDataLoader(scan_path, scan_name, skip_positions={'03_03'})
analyzer = ProjectionAnalyzer(loader)

# Test pairs of projections around 180 degrees apart
test_indices = [
    (1, steps_per_180 + 1),
    (0.5 * steps_per_180, 1.5 * steps_per_180),
    (steps_per_180, 2 * steps_per_180)
]

results = []
for idx1, idx2 in test_indices:
    shift, correlation = analyzer.find_projection_offset(idx1, idx2)
    results.append((idx1, idx2, shift, correlation))
    print(f"Projections {idx1} and {idx2}: Shift = {shift}, Correlation = {correlation:.3f}")
    fig = analyzer.visualize_alignment(idx1, idx2, shift)
    plt.show()