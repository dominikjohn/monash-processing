import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.optimize import curve_fit

def analyze_projection_overlap(proj1, proj2, max_shift=None, plot=True):
    """
    Analyzes the overlap between two CT projections using cross-correlation
    and provides visualization of the overlap region.

    Parameters:
    -----------
    proj1, proj2 : ndarray
        The two projections to analyze (1D arrays)
    max_shift : int, optional
        Maximum pixel shift to consider
    plot : bool
        Whether to show diagnostic plots

    Returns:
    --------
    dict
        Contains overlap_pixels, correlation_score, and suggested_overlap
    """
    if max_shift is None:
        max_shift = len(proj1) // 2

    # Normalize projections
    proj1_norm = (proj1 - np.mean(proj1)) / np.std(proj1)
    proj2_norm = (proj2 - np.mean(proj2)) / np.std(proj2)

    # Calculate cross-correlation
    correlation = correlate(proj1_norm, proj2_norm, mode='full')
    shifts = np.arange(-len(proj1) + 1, len(proj1))

    # Find best shift
    valid_range = (abs(shifts) <= max_shift)
    best_shift = shifts[valid_range][np.argmax(correlation[valid_range])]
    max_corr = np.max(correlation[valid_range])

    # Calculate suggested overlap
    overlap_pixels = len(proj1) - abs(best_shift)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot original projections
        ax1.plot(proj1, label='Projection 1')
        ax1.plot(proj2, label='Projection 2')
        ax1.set_title('Original Projections')
        ax1.legend()

        # Plot correlation
        ax2.plot(shifts[valid_range], correlation[valid_range])
        ax2.axvline(best_shift, color='r', linestyle='--')
        ax2.set_title(f'Cross-correlation (Best shift: {best_shift} pixels)')
        ax2.set_xlabel('Shift (pixels)')

        # Plot aligned projections
        if best_shift > 0:
            aligned_proj2 = np.pad(proj2[:-best_shift], (best_shift, 0))
        else:
            aligned_proj2 = np.pad(proj2[-best_shift:], (0, -best_shift))

        ax3.plot(proj1, label='Projection 1')
        ax3.plot(aligned_proj2, label='Projection 2 (aligned)')
        ax3.axvspan(min(len(proj1) - overlap_pixels, len(proj1)),
                    max(overlap_pixels, 0),
                    alpha=0.2, color='g', label='Overlap region')
        ax3.set_title(f'Aligned Projections (Overlap: {overlap_pixels} pixels)')
        ax3.legend()

        plt.tight_layout()
        plt.show()

    return {
        'overlap_pixels': overlap_pixels,
        'correlation_score': max_corr,
        'best_shift': best_shift
    }

def analyze_sinogram_overlap(sino1, sino2, angle_indices=None, max_shift=None):
    """
    Analyzes overlap across multiple projection angles in sinograms.

    Parameters:
    -----------
    sino1, sino2 : ndarray
        The two sinograms to analyze
    angle_indices : list, optional
        Specific angle indices to analyze
    max_shift : int, optional
        Maximum pixel shift to consider

    Returns:
    --------
    dict
        Contains average overlap and statistics
    """
    if angle_indices is None:
        # Sample a few angles across the sinogram
        angle_indices = np.linspace(0, sino1.shape[0] - 1, 5).astype(int)

    results = []
    for idx in angle_indices:
        result = analyze_projection_overlap(
            sino1[idx], sino2[idx],
            max_shift=max_shift, plot=False
        )
        results.append(result)

    # Aggregate results
    overlaps = [r['overlap_pixels'] for r in results]
    correlations = [r['correlation_score'] for r in results]

    # Plot summary
    plt.figure(figsize=(10, 6))
    plt.plot(angle_indices, overlaps, 'o-')
    plt.xlabel('Projection Index')
    plt.ylabel('Overlap (pixels)')
    plt.title(f'Overlap vs Projection Angle\nMean: {np.mean(overlaps):.1f} ± {np.std(overlaps):.1f} pixels')
    plt.grid(True)
    plt.show()

    return {
        'mean_overlap': np.mean(overlaps),
        'std_overlap': np.std(overlaps),
        'mean_correlation': np.mean(correlations),
        'overlap_by_angle': dict(zip(angle_indices, overlaps))
    }