import numpy as np
from scipy.signal import correlate
from typing import Tuple
from pathlib import Path
from monash_processing.core.multi_position_data_loader import MultiPositionDataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def find_projection_offset(data_loader, angle_index1: int, angle_index2: int) -> Tuple[int, float]:
    """
    Find the horizontal shift between two projections using cross-correlation.

    Args:
        data_loader: DataLoader instance
        angle_index1: Index of first projection
        angle_index2: Index of second projection (typically ~180 degrees apart)

    Returns:
        Tuple of (optimal_shift, correlation_score)
    """
    # Load the two projections
    proj1 = data_loader.load_processed_projection(angle_index1, 'dx')
    proj2 = data_loader.load_processed_projection(angle_index2, 'dx')

    # Remove batch dimension if present
    if proj1.ndim > 2:
        proj1 = proj1.squeeze()
    if proj2.ndim > 2:
        proj2 = proj2.squeeze()

    # Flip the second projection horizontally (since it's ~180 degrees rotated)
    proj2_flipped = np.fliplr(proj2)

    # Take multiple horizontal rows around the center
    center_y = proj1.shape[0] // 2
    num_rows = 10
    rows_to_check = np.linspace(
        center_y - proj1.shape[0] // 4,
        center_y + proj1.shape[0] // 4,
        num_rows,
        dtype=int
    )

    # Store shifts from each row pair
    shifts = []
    correlations = []

    # Correlate corresponding rows to find horizontal shifts
    for y in rows_to_check:
        row1 = proj1[y, :]
        row2 = -proj2_flipped[y, :]

        # Correlate these 1D arrays to find horizontal shift
        correlation = correlate(row1, row2, mode='full')
        shift = correlation.argmax() - (len(row1) - 1)

        # Only keep shifts with good correlation
        max_corr = correlation.max()
        if max_corr > 0.5 * np.sqrt(np.sum(row1 ** 2) * np.sum(row2 ** 2)):  # Correlation threshold
            shifts.append(shift)
            correlations.append(max_corr)

    if not shifts:
        return 0, 0.0  # Return no shift if no good correlations found

    # Use median shift as the result (more robust than mean)
    optimal_shift = int(np.median(shifts))
    max_correlation = np.mean(correlations)

    return optimal_shift, max_correlation

# Set your parameters
scan_path = Path("/data/mct/22203/")
scan_name = "K21N_sample"
pixel_size = 1.444e-6 # m

# Calculate indices for opposing projections (180 degrees apart)
total_angles = 3640
angle_per_step = 364 / total_angles
steps_per_180 = int(180 / angle_per_step)

loader = MultiPositionDataLoader(scan_path, scan_name)

# Test a few pairs of projections around 180 degrees apart
test_indices = [(1, steps_per_180+1),
                #(steps_per_180 // 2, 3 * steps_per_180 // 2),
                #(steps_per_180, 2 * steps_per_180)
                ]


def visualize_projection_shift(data_loader, angle_index1: int, angle_index2: int, shift: int):
    """
    Visualize the alignment of phase contrast projections with proper horizontal shift handling.
    """
    # Load projections
    proj1 = data_loader.load_processed_projection(angle_index1, 'dx')
    proj2 = data_loader.load_processed_projection(angle_index2, 'dx')

    # Remove batch dimension if present
    if proj1.ndim > 2:
        proj1 = proj1.squeeze()
    if proj2.ndim > 2:
        proj2 = proj2.squeeze()

    # Print data ranges
    print(f"Projection 1 range: {proj1.min():.3f} to {proj1.max():.3f}")
    print(f"Projection 2 range: {proj2.min():.3f} to {proj2.max():.3f}")

    # Flip the second projection horizontally and in sign
    proj2_flipped = -np.fliplr(proj2)

    # Calculate the full width needed for shifted images
    full_width = proj1.shape[1] + abs(shift)

    # Create empty arrays for the aligned projections
    proj1_aligned = np.full((proj1.shape[0], full_width), np.nan)
    proj2_aligned = np.full((proj2.shape[0], full_width), np.nan)

    # Position the projections in the wider frame
    if shift >= 0:
        # Proj2 starts at left
        proj2_aligned[:, :proj2_flipped.shape[1]] = proj2_flipped
        # Proj1 starts after abs(shift)
        proj1_aligned[:, abs(shift):abs(shift) + proj1.shape[1]] = proj1
    else:
        # Proj1 starts at left
        proj1_aligned[:, :proj1.shape[1]] = proj1
        # Proj2 starts after shift
        proj2_aligned[:, shift:shift + proj2_flipped.shape[1]] = proj2_flipped


    # Create composite image
    composite = np.zeros_like(proj1_aligned)
    overlap_mask = ~(np.isnan(proj1_aligned) | np.isnan(proj2_aligned))

    # Fill in non-overlapping regions
    composite[~np.isnan(proj1_aligned)] = proj1_aligned[~np.isnan(proj1_aligned)]
    composite[~np.isnan(proj2_aligned)] = proj2_aligned[~np.isnan(proj2_aligned)]

    # Average in overlapping regions
    composite[overlap_mask] = (proj1_aligned[overlap_mask] + proj2_aligned[overlap_mask]) / 2

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f'Projection Alignment Visualization\nAngle indices: {angle_index1} and {angle_index2}, Shift: {shift} pixels')

    # Plot original projections
    im1 = axes[0, 0].imshow(proj1, cmap='gray', vmin=-0.6, vmax=0.2)
    axes[0, 0].set_title('Projection 1')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(proj2_flipped, cmap='gray', vmin=-0.6, vmax=0.2)
    axes[0, 1].set_title('Projection 2 (Flipped)')
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot aligned projections with overlap
    im3 = axes[1, 0].imshow(proj1_aligned, cmap='gray', vmin=-0.6, vmax=0.2)
    axes[1, 0].set_title('Projection 1 (Aligned)')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(composite, cmap='gray', vmin=-0.6, vmax=0.2)
    axes[1, 1].set_title('Composite (Averaged in Overlap)')
    plt.colorbar(im4, ax=axes[1, 1])

    # Add overlap region information
    overlap_pixels = np.sum(overlap_mask)
    overlap_percentage = (overlap_pixels / (proj1.shape[0] * proj1.shape[1])) * 100
    fig.text(0.02, 0.02,
             f'Overlap region: {overlap_pixels} pixels ({overlap_percentage:.1f}% of original size)\n'
             f'Total width: {full_width} pixels',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


results = []
for idx1, idx2 in test_indices:
    shift, correlation = find_projection_offset(loader, idx1, idx2)
    results.append((idx1, idx2, shift, correlation))
    print(f"Projections {idx1} and {idx2}: Shift = {shift}, Correlation = {correlation:.3f}")
    fig = visualize_projection_shift(loader, idx1, idx2, shift)
    plt.show()

# Find the most consistent shift
shifts = [r[2] for r in results]
median_shift = int(np.median(shifts))
print(f"\nRecommended shift: {median_shift} pixels")