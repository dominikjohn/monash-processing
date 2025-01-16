import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.measure import block_reduce
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

def plot_slice(data, slice_idx, pixel_size,
               cmap='grey',
               title=None,
               vmin=None,
               vmax=None,
               figsize=(10, 8),
               fontsize=16):
    """
    Plot a slice with physical units based on pixel size.

    Parameters:
    -----------
    data : numpy.ndarray
        3D array of data
    slice_idx : int
        Index of the slice to plot
    pixel_size : float
        Size of each pixel in meters (e.g., 1.444e-6 * 4 for 4x binned data)
    fontsize : int
        Font size for all text elements (default: 16)
    """
    # Set the font size globally
    plt.rcParams.update({'font.size': fontsize})

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get the slice
    slice_data = data[slice_idx] if len(data.shape) == 3 else data

    # Plot the image
    im = ax.imshow(slice_data,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)

    # Add scalebar
    scalebar = ScaleBar(pixel_size,  # meters per pixel
                        "m",  # meter unit
                        length_fraction=.2,
                        color='white',
                        box_alpha=0,
                        location='lower right',
                        font_properties={'size': fontsize})
    ax.add_artist(scalebar)

    # Add colorbar with matching height and title
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)  # Increased pad from 0.05 to 0.15
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'Electron density $\rho_e$ [1/nm$^3$]', size=fontsize, labelpad=15)
    #cbar.set_label(r'Attenuation coefficient $\mu$ [1/cm]', size=fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=fontsize)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    return fig, ax

binning_factor = 4
psize = 1.444e-6 * binning_factor  # 4x binned data

binned_volume = block_reduce(edensity_volume, (binning_factor, 1, 1), np.mean)

calibration = .8567
binned_volume *= calibration

# Example with your data:
fig, ax = plot_slice(binned_volume.transpose(1, 0, 2),
                     slice_idx=290,
                     pixel_size=psize,
                     vmin=0,
                     vmax=650,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
plt.savefig(os.path.join(home_dir, f'slice_plot.png'), dpi=1200, bbox_inches='tight')
plt.show()