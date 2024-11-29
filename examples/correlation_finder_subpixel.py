import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from monash_processing.core.data_loader import DataLoader
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask
from pathlib import Path


def shift_array(arr, shift):
    """
    Shift array by a floating-point number using interpolation.

    Args:
        arr: Input array
        shift: Floating-point shift amount

    Returns:
        Shifted array using cubic interpolation
    """
    return ndimage.shift(arr, (0, shift), order=3, mode='constant', cval=np.nan)


def find_optimal_shift(proj1, proj2_flipped, min_shift=800, max_shift=1400, step=0.05):
    """
    Find the optimal shift that maximizes correlation between overlapping regions.

    Args:
        proj1: First projection
        proj2_flipped: Second projection (flipped)
        min_shift: Minimum shift to try
        max_shift: Maximum shift to try
        step: Step size for shift values (default: 0.05 for subpixel precision)

    Returns:
        tuple: (optimal_shift, correlation_scores, shifts_tested)
    """
    shifts = np.arange(min_shift, max_shift + step, step)
    correlation_scores = []

    for shift in shifts:
        # Create aligned arrays with current shift
        full_width = int(proj1.shape[1] + abs(shift)) + 2  # +2 for interpolation buffer
        p1 = np.full((proj1.shape[0], full_width), np.nan)
        p2 = np.full((proj1.shape[0], full_width), np.nan)

        if shift >= 0:
            # Use interpolation for subpixel shifts
            shifted_proj2 = shift_array(proj2_flipped, 0)  # No shift for proj2
            shifted_proj1 = shift_array(proj1, shift)  # Subpixel shift for proj1

            p2[:, :proj2_flipped.shape[1]] = shifted_proj2
            p1[:, int(shift):int(shift) + proj1.shape[1]] = shifted_proj1
        else:
            shifted_proj1 = shift_array(proj1, 0)  # No shift for proj1
            shifted_proj2 = shift_array(proj2_flipped, -shift)  # Subpixel shift for proj2

            p1[:, :proj1.shape[1]] = shifted_proj1
            p2[:, int(-shift):int(-shift) + proj2_flipped.shape[1]] = shifted_proj2

        # Find overlap region
        overlap = ~(np.isnan(p1) | np.isnan(p2))
        if np.any(overlap):
            # Extract overlapping regions
            p1_overlap = p1[overlap].reshape(-1)
            p2_overlap = p2[overlap].reshape(-1)

            # Calculate correlation coefficient
            corr = np.corrcoef(p1_overlap, p2_overlap)[0, 1]
            correlation_scores.append(corr)
        else:
            correlation_scores.append(-1)  # No overlap

    optimal_shift = shifts[np.argmax(correlation_scores)]
    return optimal_shift, correlation_scores, list(shifts)


def create_interactive_blend_viewer_with_optimization(data_loader, angle_index1, angle_index2):
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 8))
    ax_image = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_corr = plt.subplot2grid((3, 1), (2, 0))
    plt.subplots_adjust(bottom=0.2, hspace=0.3)

    def load_proj(idx):
        return data_loader.load_processed_projection(idx, 'dx')

    # Load and prepare base images once
    proj1_raw = BadPixelMask.correct_bad_pixels(load_proj(angle_index1))[0]
    proj2_raw = BadPixelMask.correct_bad_pixels(load_proj(angle_index2))[0]

    # Print raw statistics for debugging
    print(f"Raw proj1 stats - min: {np.min(proj1_raw)}, max: {np.max(proj1_raw)}, mean: {np.mean(proj1_raw)}")
    print(f"Raw proj2 stats - min: {np.min(proj2_raw)}, max: {np.max(proj2_raw)}, mean: {np.mean(proj2_raw)}")

    # Normalize proj1 based on rightmost region
    right_mean1 = np.mean(proj1_raw[:, -100:-5])
    proj1 = proj1_raw - right_mean1

    # Flip proj2 (horizontal flip and negate values) then normalize
    proj2_flipped_raw = np.fliplr(proj2_raw)  # First flip
    proj2_flipped_raw = -proj2_flipped_raw    # Then negate
    left_mean2 = np.mean(proj2_flipped_raw[:, 5:100])
    proj2_flipped = proj2_flipped_raw - left_mean2

    # Print normalized statistics for debugging
    print(f"Normalized proj1 stats - min: {np.min(proj1)}, max: {np.max(proj1)}, mean: {np.mean(proj1)}")
    print(f"Normalized proj2_flipped stats - min: {np.min(proj2_flipped)}, max: {np.max(proj2_flipped)}, mean: {np.mean(proj2_flipped)}")

    # Verify the projections are different
    print(f"Are projections identical? {np.allclose(proj1, proj2_flipped)}")
    print(f"Correlation between projections: {np.corrcoef(proj1.flatten(), proj2_flipped.flatten())[0,1]}")

    # Create slider and optimize button axes
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_button = plt.axes([0.8, 0.1, 0.1, 0.03])

    slider = Slider(
        ax=ax_slider,
        label='Shift',
        valmin=1200,
        valmax=1400,
        valinit=1200,
        valstep=50,  # Set step size to 0.05 for subpixel precision
    )

    optimize_button = Button(ax_button, 'Optimize')

    # Initial plots
    img_plot = None
    corr_plot = None
    optimal_shift = None
    correlation_scores = None
    shifts_tested = None

    def update(val, show_correlation=False):
        nonlocal img_plot, corr_plot
        # Clear previous plots
        if img_plot is not None:
            img_plot.remove()
        if corr_plot is not None:
            for artist in ax_corr.lines + ax_corr.collections:
                artist.remove()

        # Get current shift value
        shift = slider.val  # Now using float value directly

        # Create aligned arrays with current shift
        full_width = int(proj1.shape[1] + abs(shift)) + 2  # +2 for interpolation buffer
        p1 = np.full((proj1.shape[0], full_width), np.nan)
        p2 = np.full((proj1.shape[0], full_width), np.nan)

        # Debug prints
        print(f"Shift value: {shift}")
        print(f"Proj1 shape: {proj1.shape}")
        print(f"Proj2_flipped shape: {proj2_flipped.shape}")
        print(f"Full width: {full_width}")

        # Ensure projections are different
        print(f"Proj1 mean: {np.mean(proj1)}")
        print(f"Proj2_flipped mean: {np.mean(proj2_flipped)}")

        if shift >= 0:
            # Keep proj2_flipped at the start
            p2[:, :proj2_flipped.shape[1]] = proj2_flipped
            # Shift proj1 by the specified amount using interpolation
            shifted_proj1 = shift_array(proj1, shift)
            start_idx = int(shift)
            p1[:, start_idx:start_idx + proj1.shape[1]] = shifted_proj1
        else:
            # Keep proj1 at the start
            p1[:, :proj1.shape[1]] = proj1
            # Shift proj2_flipped by the specified amount using interpolation
            shifted_proj2 = shift_array(proj2_flipped, -shift)
            start_idx = int(-shift)
            p2[:, start_idx:start_idx + proj2_flipped.shape[1]] = shifted_proj2

        # Find overlap region and blend
        overlap = ~(np.isnan(p1) | np.isnan(p2))
        cols = np.where(np.any(overlap, axis=0))[0]

        if len(cols):
            comp = np.full_like(p1, np.nan)
            blend_start = cols[0]
            blend_end = cols[-1]
            blend_width = blend_end - blend_start
            x = np.linspace(0, 1, blend_width)
            blend_region = (slice(None), slice(blend_start, blend_end))
            comp[blend_region] = (p1[blend_region] * x[None, :] +
                                p2[blend_region] * (1 - x[None, :]))
            comp = np.where(np.isnan(comp), p1, comp)
            comp = np.where(np.isnan(comp), p2, comp)
        else:
            comp = np.where(np.isnan(p1), p2, p1)

        # Update image plot
        ax_image.clear()
        img_plot = ax_image.imshow(comp, cmap='gray')
        ax_image.set_title(f'Blended Image (Shift: {shift:.2f}px)')

        # Update correlation plot if available
        if show_correlation and correlation_scores is not None:
            ax_corr.clear()
            ax_corr.plot(shifts_tested, correlation_scores, 'b-', label='Correlation Score')
            ax_corr.axvline(x=shift, color='r', linestyle='--', label='Current Shift')
            if optimal_shift is not None:
                ax_corr.axvline(x=optimal_shift, color='g', linestyle='--', label='Optimal Shift')
            ax_corr.set_xlabel('Shift (pixels)')
            ax_corr.set_ylabel('Correlation')
            ax_corr.legend()
            ax_corr.grid(True)

        fig.canvas.draw_idle()

    def optimize(_):
        nonlocal optimal_shift, correlation_scores, shifts_tested
        print("Finding optimal shift...")
        optimal_shift, correlation_scores, shifts_tested = find_optimal_shift(
            proj1, proj2_flipped,
            int(slider.valmin),
            int(slider.valmax),
            step=0.05  # Set step size for optimization
        )
        print(f"Optimal shift found: {optimal_shift:.2f}px")

        # Animate slider to optimal value
        current = slider.val
        steps = np.linspace(current, optimal_shift, 50)
        for step in steps:
            slider.set_val(step)
            update(step, show_correlation=True)
            plt.pause(0.01)

        # Final update with correlation plot
        update(optimal_shift, show_correlation=True)

    # Connect the update function to the slider
    slider.on_changed(lambda val: update(val, show_correlation=True if correlation_scores is not None else False))
    optimize_button.on_clicked(optimize)

    # Initial update
    update(1200)

    plt.show()


# Usage
data_loader = DataLoader(Path("/data/mct/22203/"), "K3_2E")
create_interactive_blend_viewer_with_optimization(data_loader, 800, 2600)