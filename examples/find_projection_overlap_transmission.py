import numpy as np
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask

from monash_processing.core.data_loader import DataLoader
import matplotlib
from monash_processing.utils.ImageViewer import ImageViewer as imshow
from pathlib import Path

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def prepare_and_align_projections(data_loader, angle_index1: int, angle_index2: int, shift: int):
    """
    Prepare and align two projections with normalization and linear blending.

    Args:
        data_loader: DataLoader instance
        angle_index1: Index of first projection
        angle_index2: Index of second projection
        shift: Horizontal shift in pixels

    Returns:
        tuple: (composite image, blend_info dictionary)
    """

    def load_proj(idx):
        return data_loader.load_processed_projection(idx, 'dx')

    # Load and prepare images
    proj1_raw = BadPixelMask.correct_bad_pixels(load_proj(angle_index1))[0]
    proj2_raw = BadPixelMask.correct_bad_pixels(load_proj(angle_index2))[0]

    # Normalize proj1 based on rightmost region
    right_mean1 = np.mean(proj1_raw[:, -100:-5])
    proj1 = proj1_raw - right_mean1

    # Flip and normalize proj2 based on leftmost region
    proj2_flipped_raw = -np.fliplr(proj2_raw)
    left_mean2 = np.mean(proj2_flipped_raw[:, 5:100])
    proj2_flipped = proj2_flipped_raw - left_mean2

    # Create aligned arrays
    full_width = proj1.shape[1] + abs(shift)
    p1 = np.full((proj1.shape[0], full_width), np.nan)
    p2 = np.full((proj1.shape[0], full_width), np.nan)

    if shift >= 0:
        p2[:, :proj2_flipped.shape[1]] = proj2_flipped
        p1[:, shift:shift + proj1.shape[1]] = proj1
    else:
        p1[:, :proj1.shape[1]] = proj1
        p2[:, -shift:-shift + proj2_flipped.shape[1]] = proj2_flipped

    # Find overlap region
    overlap = ~(np.isnan(p1) | np.isnan(p2))
    cols = np.where(np.any(overlap, axis=0))[0]

    if len(cols):
        # Initialize composite
        comp = np.full_like(p1, np.nan)

        # Use entire overlap region for blending
        blend_start = cols[0]
        blend_end = cols[-1]
        blend_width = blend_end - blend_start

        # Create linear blend weights for the entire overlap region
        x = np.linspace(0, 1, blend_width)
        blend_region = (slice(None), slice(blend_start, blend_end))

        # Linear blend over the entire overlap region
        comp[blend_region] = (p1[blend_region] * x[None, :] +
                              p2[blend_region] * (1 - x[None, :]))

        # Fill non-overlapping regions
        comp = np.where(np.isnan(comp), p1, comp)
        comp = np.where(np.isnan(comp), p2, comp)
    else:
        comp = np.where(np.isnan(p1), p2, p1)
        blend_start = blend_end = blend_width = None

    blend_info = {
        'blend_start': blend_start,
        'blend_end': blend_end,
        'blend_width': blend_width,
        'total_width': full_width,
        'overlap_pixels': np.sum(overlap),
        'overlap_percentage': (np.sum(overlap) / proj1.size) * 100,
        'proj1': proj1,
        'proj2_flipped': proj2_flipped
    }

    return comp, blend_info

# Usage
data_loader = DataLoader(Path("/data/mct/22203/"), "K3_2E")
comp, info = prepare_and_align_projections(data_loader, 600, 2400, 1200)