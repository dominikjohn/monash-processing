import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask


def interactive_projection_alignment(data_loader, angle_index1: int, angle_index2: int):
    # Load and prepare images
    def load_proj(idx):
        return data_loader.load_processed_projection(idx, 'dx')

    proj2, proj1 = BadPixelMask.correct_bad_pixels(load_proj(angle_index1))[0], \
    BadPixelMask.correct_bad_pixels(load_proj(angle_index2))[0]
    proj2_flipped = -np.fliplr(proj2)

    # Setup figure and axes
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(4, 2, height_ratios=[0.1, 0.1, 1, 1])
    axes = [fig.add_subplot(gs[i, j]) for i, j in [(2, 0), (2, 1), (3, 0), (3, 1)]]
    slider_ax, button_ax = fig.add_subplot(gs[0, :]), plt.subplot(gs[1, :])
    button_ax.set_visible(False)

    # Create buttons
    btns = [Button(plt.axes([x, 0.85, 0.05, 0.03]), s) for x, s in [(0.4, '-'), (0.55, '+')]]

    # Show original images
    [ax.imshow(img, cmap='gray', vmin=-0.6, vmax=0.6) for ax, img in zip(axes[:2], [proj1, proj2_flipped])]
    axes[0].set_title('Projection 1')
    axes[1].set_title('Projection 2 (Flipped)')

    def update_alignment(shift):
        [ax.clear() for ax in axes[2:]]
        shift = int(shift)
        full_width = proj1.shape[1] + abs(shift)

        # Create aligned arrays
        p1 = np.full((proj1.shape[0], full_width), np.nan)
        p2 = np.full((proj1.shape[0], full_width), np.nan)
        if shift >= 0:
            p2[:, :proj2_flipped.shape[1]] = proj2_flipped
            p1[:, shift:shift + proj1.shape[1]] = proj1
        else:
            p1[:, :proj1.shape[1]] = proj1
            p2[:, -shift:-shift + proj2_flipped.shape[1]] = proj2_flipped

        # Find overlap and create composite
        overlap = ~(np.isnan(p1) | np.isnan(p2))
        cols = np.where(np.any(overlap, axis=0))[0]

        if len(cols):
            # Initialize composite
            comp = np.full_like(p1, np.nan)

            # Find overlap center and create blend region
            blend_width = 400
            overlap_center = (cols[0] + cols[-1]) // 2
            blend_start = overlap_center - blend_width // 2
            blend_end = overlap_center + blend_width // 2

            print('Mean P2 in overlap', np.mean(p2[:, blend_start:blend_end]))
            print('Mean P1 in overlap', np.mean(p1[:, blend_start:blend_end]))
            diff = np.mean(p1[:, blend_start:blend_end]) - np.mean(p2[:, blend_start:blend_end])



            p2 += diff

            print('Mean P2 now', np.mean(p2[:, blend_start:blend_end]))

            # Use proj2 before blend, proj1 after blend
            comp[:, :blend_start] = p2[:, :blend_start]
            comp[:, blend_end:] = p1[:, blend_end:]

            # Create blend weights for transition region
            x = np.linspace(0, 1, blend_width)
            blend_region = (slice(None), slice(blend_start, blend_end))
            comp[blend_region] = (p1[blend_region] * x[None, :] +
                                  p2[blend_region] * (1 - x[None, :]))

            # Fill any remaining gaps
            comp = np.where(np.isnan(comp), p1, comp)
            comp = np.where(np.isnan(comp), p2, comp)

            # Show composite and zoom
            axes[2].imshow(comp, cmap='gray', vmin=-0.6, vmax=0.6)

            # Setup zoom around blend region
            zoom_y = proj1.shape[0] // 2
            z_start = max(0, overlap_center - 200)
            z_end = min(full_width, overlap_center + 200)
            y_start = max(0, zoom_y - 100)
            y_end = min(proj1.shape[0], zoom_y + 100)

            axes[2].add_patch(plt.Rectangle((z_start, y_start), z_end - z_start,
                                            y_end - y_start, fill=False, color='r', linestyle=':'))
            axes[3].imshow(comp[y_start:y_end, z_start:z_end], cmap='gray', vmin=-0.6, vmax=0.6)

            # Add blend region indicators
            #axes[2].axvline(x=blend_start, color='g', linestyle=':', alpha=0.5)
            #axes[2].axvline(x=blend_end, color='g', linestyle=':', alpha=0.5)
            if z_start <= blend_start <= z_end:
                axes[3].axvline(x=blend_start - z_start, color='g', linestyle=':', alpha=0.5)
            if z_start <= blend_end <= z_end:
                axes[3].axvline(x=blend_end - z_start, color='g', linestyle=':', alpha=0.5)

        else:
            comp = np.where(np.isnan(p1), p2, p1)
            axes[2].imshow(comp, cmap='gray', vmin=-0.6, vmax=0.6)
            axes[3].text(0.5, 0.5, 'No overlap', ha='center', va='center', transform=axes[3].transAxes)

        axes[2].set_title('Composite (10px Blend)')
        axes[3].set_title('Zoomed Composite')

        if hasattr(fig, 'text_artist'): fig.text_artist.remove()
        fig.text_artist = fig.text(0.02, 0.02,
                                   f'Shift: {shift} px\nOverlap: {np.sum(overlap)} px ({(np.sum(overlap) / proj1.size) * 100:.1f}%)\n' +
                                   f'Blend width: {blend_width} px\nTotal width: {full_width} px',
                                   bbox=dict(facecolor='white', alpha=0.8))
        fig.canvas.draw_idle()

    # Setup controls
    slider = Slider(slider_ax, 'Horizontal Shift', 1200, 1220, valinit=804, valstep=1)
    slider.on_changed(update_alignment)

    def update(val):
        slider.set_val(min(max(slider.val + val, slider.valmin), slider.valmax))

    [btn.on_clicked(lambda _: update(v)) for btn, v in zip(btns, [-1, 1])]
    fig.canvas.mpl_connect('key_press_event',
                           lambda e: update(1 if e.key in ['=', '+'] else -1 if e.key in ['-', '_'] else 0))

    update_alignment(804)
    plt.tight_layout()
    plt.show()
    return fig, slider

data_loader = DataLoader(Path("/data/mct/22203/"), "K3_2E")
def load_proj(idx):
    return data_loader.load_processed_projection(idx, 'dx')

#proj1, proj2 = BadPixelMask.correct_bad_pixels(load_proj(1))[0], \
#    BadPixelMask.correct_bad_pixels(load_proj(1+1800))[0]
#proj2_flipped = -np.fliplr(proj2)


# Usage
fig, slider = interactive_projection_alignment(data_loader, 800, 2600)