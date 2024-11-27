import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from monash_processing.core.multi_position_data_loader import MultiPositionDataLoader

def interactive_projection_alignment(data_loader, angle_index1: int, angle_index2: int):
    """Create interactive visualization of projection alignment with composite zoom."""

    # Load and correct projections
    def load_proj(idx):
        proj = data_loader.load_projections(idx)
        ff = data_loader.load_flat_fields()
        dc = data_loader.load_flat_fields(dark=True)
        return np.mean(data_loader.perform_flatfield_correction(proj, ff, dc), axis=0)

    proj1, proj2 = load_proj(angle_index1), load_proj(angle_index2)
    proj2_flipped = np.fliplr(proj2)

    # Setup figure
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(4, 2, height_ratios=[0.1, 0.1, 1, 1])

    # Create slider and button axes
    slider_ax = fig.add_subplot(gs[0, :])
    button_ax = plt.subplot(gs[1, :])
    button_ax.set_visible(False)

    # Create button axes
    minus_button_ax = plt.axes([0.4, 0.85, 0.05, 0.03])
    plus_button_ax = plt.axes([0.55, 0.85, 0.05, 0.03])

    # Create main axes
    ax_orig1 = fig.add_subplot(gs[2, 0])
    ax_orig2 = fig.add_subplot(gs[2, 1])
    ax_comp = fig.add_subplot(gs[3, 0])
    ax_zoom = fig.add_subplot(gs[3, 1])

    initial_shift = 1325

    # Show original images
    ax_orig1.imshow(proj1, cmap='gray')
    ax_orig1.set_title('Projection 1')
    ax_orig2.imshow(proj2_flipped, cmap='gray')
    ax_orig2.set_title('Projection 2 (Flipped)')

    def update_alignment(shift):
        """Update visualization with new shift and zoomed views"""
        ax_comp.clear()
        ax_zoom.clear()

        # Align projections
        shift = int(shift)
        full_width = proj1.shape[1] + abs(shift)
        proj1_aligned = np.full((proj1.shape[0], full_width), np.nan)
        proj2_aligned = np.full((proj2.shape[0], full_width), np.nan)

        if shift >= 0:
            proj2_aligned[:, :proj2_flipped.shape[1]] = proj2_flipped
            proj1_aligned[:, shift:shift + proj1.shape[1]] = proj1
        else:
            proj1_aligned[:, :proj1.shape[1]] = proj1
            proj2_aligned[:, -shift:-shift + proj2_flipped.shape[1]] = proj2_flipped

        # Create blended composite
        overlap_mask = ~(np.isnan(proj1_aligned) | np.isnan(proj2_aligned))
        overlap_cols = np.where(np.any(overlap_mask, axis=0))[0]

        if len(overlap_cols) > 0:
            # Create weight arrays for blending
            weights = np.zeros_like(proj1_aligned)
            overlap_start = overlap_cols[0]
            overlap_end = overlap_cols[-1]
            overlap_width = overlap_end - overlap_start

            # Create linear weight transition
            for i in range(overlap_width):
                weight = i / overlap_width
                weights[:, overlap_start + i] = weight

            # Initialize composite with proj2_aligned
            composite = np.copy(proj2_aligned)

            # Blend in the overlap region
            overlap_region = ~np.isnan(proj1_aligned) & ~np.isnan(proj2_aligned)
            composite[overlap_region] = (
                    proj1_aligned[overlap_region] * weights[overlap_region] +
                    proj2_aligned[overlap_region] * (1 - weights[overlap_region])
            )

            # Fill in the non-overlapping regions from proj1
            composite = np.where(np.isnan(composite), proj1_aligned, composite)

            # Show composite
            ax_comp.imshow(composite, cmap='gray')
            ax_comp.set_title('Composite (Blended)')

            # Define and show zoom region
            zoom_height = 300
            zoom_width = 400

            zoom_center_y = proj1.shape[0] // 2
            zoom_center_x = (overlap_start + overlap_end) // 2

            zoom_center_x -= 600
            zoom_center_y += 600

            # Ensure zoom region stays within bounds
            zoom_start = max(0, zoom_center_x - zoom_width // 2)
            zoom_end = min(full_width, zoom_center_x + zoom_width // 2)
            y_start = max(0, zoom_center_y - zoom_height // 2)
            y_end = min(proj1.shape[0], zoom_center_y + zoom_height // 2)

            # Show zoom region
            ax_zoom.imshow(composite[y_start:y_end, zoom_start:zoom_end], cmap='gray')
            ax_zoom.set_title('Zoomed Composite')

            # Add zoom region indicator
            ax_comp.add_patch(plt.Rectangle((zoom_start, y_start),
                                            zoom_end - zoom_start,
                                            y_end - y_start,
                                            fill=False, color='r', linestyle=':'))

        else:
            composite = np.where(np.isnan(proj1_aligned), proj2_aligned, proj1_aligned)
            ax_comp.imshow(composite, cmap='gray')
            ax_comp.set_title('Composite')
            ax_zoom.text(0.5, 0.5, 'No overlap region to zoom',
                         ha='center', va='center', transform=ax_zoom.transAxes)

        # Update stats
        if hasattr(fig, 'text_artist'):
            fig.text_artist.remove()
        overlap_pixels = np.sum(overlap_mask)
        fig.text_artist = fig.text(0.02, 0.02,
                                   f'Shift: {shift} pixels\n'
                                   f'Overlap: {overlap_pixels} pixels ({(overlap_pixels / (proj1.size)) * 100:.1f}%)\n'
                                   f'Overlap width: {len(overlap_cols)} pixels\n'
                                   f'Total width: {full_width} pixels',
                                   bbox=dict(facecolor='white', alpha=0.8))

        fig.canvas.draw_idle()

    # Create slider and buttons
    slider = Slider(slider_ax, 'Horizontal Shift', 1000, 2000, valinit=initial_shift, valstep=1)
    minus_button = Button(minus_button_ax, '-')
    plus_button = Button(plus_button_ax, '+')

    def on_key_press(event):
        if event.key == '+' or event.key == '=':
            new_val = min(slider.valmax, slider.val + 1)
            slider.set_val(new_val)
        elif event.key == '-' or event.key == '_':
            new_val = max(slider.valmin, slider.val - 1)
            slider.set_val(new_val)

    def on_minus_button(event):
        new_val = max(slider.valmin, slider.val - 1)
        slider.set_val(new_val)

    def on_plus_button(event):
        new_val = min(slider.valmax, slider.val + 1)
        slider.set_val(new_val)

    # Connect all events
    slider.on_changed(update_alignment)
    minus_button.on_clicked(on_minus_button)
    plus_button.on_clicked(on_plus_button)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Initial update
    update_alignment(initial_shift)

    plt.tight_layout()
    plt.show()
    return fig, slider


# Usage
scan_path = Path("/data/mct/22203/")
scan_name = "K21N_sample"
loader = MultiPositionDataLoader(scan_path, scan_name, skip_positions={'03_03'})
fig, slider = interactive_projection_alignment(loader, 800, 2600)