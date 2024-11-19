import numpy as np
from scipy.signal import correlate
from typing import Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from monash-processing.core.data_loader import ScanDataLoader

def find_projection_offset(data_loader, flat_field, dark_current, angle_index1: int, angle_index2: int) -> Tuple[int, float]:
    """
    Find the vertical shift between two projections using cross-correlation.

    Args:
        data_loader: DataLoader instance
        flat_field: Flat field image
        dark_current: Dark current image
        angle_index1: Index of first projection
        angle_index2: Index of second projection (typically ~180 degrees apart)

    Returns:
        Tuple of (optimal_shift, correlation_score)
    """

    # Load the two projections
    proj1 = data_loader.load_projections(angle_index1)
    proj2 = data_loader.load_projections(angle_index2)

    # Perform flatfield correction
    proj1 = data_loader.perform_flatfield_correction(proj1, flat_field, dark_current)
    proj2 = data_loader.perform_flatfield_correction(proj2, flat_field, dark_current)

    # Remove batch dimension if present for some reason
    if proj1.ndim > 2:
        proj1 = proj1.squeeze()
    if proj2.ndim > 2:
        proj2 = proj2.squeeze()

    # Flip the second projection horizontally (since it's ~180 degrees rotated)
    proj2_flipped = np.fliplr(proj2)

    # Take multiple slices around the center
    center_x = proj1.shape[0] // 2
    num_cols = 10
    start_y = center_x - num_cols // 2
    end_y = center_x + num_cols // 2

    # Store shifts from each column pair
    shifts = []
    correlations = []

    # Correlate corresponding columns to find vertical shifts
    for y in range(start_y, end_y):
        col1 = proj1[y, :]
        col2 = proj2_flipped[y, :]

        # Correlate these 1D arrays to find vertical shift
        correlation = correlate(col1, col2, mode='full')
        shift = correlation.argmax() - (len(col1) - 1)

        shifts.append(shift)
        correlations.append(correlation.max())

    # Use median shift as the result (more robust than mean)
    optimal_shift = int(np.median(shifts))
    max_correlation = np.mean(correlations)

    return optimal_shift, max_correlation

# Set your parameters
scan_path = Path("/path/to/scan")
scan_name = "P6_ReverseOrder"
pixel_size = 1.444e-6 # m

# Calculate indices for opposing projections (180 degrees apart)
total_angles = 3640
angle_per_step = 364 / total_angles
steps_per_180 = int(180 / angle_per_step)

loader = ScanDataLoader(scan_path, scan_name)

# We are only using step 0 for matching -> Shape: X, Y
flat_fields_step0 = loader.load_flat_fields()[0]
dark_current = loader.load_flat_fields(dark=True) # Shape: X, Y

# Test a few pairs of projections around 180 degrees apart
test_indices = [(0, steps_per_180),
                (steps_per_180 // 2, 3 * steps_per_180 // 2),
                (steps_per_180, 2 * steps_per_180)]

results = []
for idx1, idx2 in test_indices:
    shift, correlation = find_projection_offset(loader, flat_fields_step0, dark_current, idx1, idx2)
    results.append((idx1, idx2, shift, correlation))
    print(f"Projections {idx1} and {idx2}: Shift = {shift}, Correlation = {correlation:.3f}")

# Find the most consistent shift
shifts = [r[2] for r in results]
median_shift = int(np.median(shifts))
print(f"\nRecommended shift: {median_shift} pixels")