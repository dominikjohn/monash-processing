import tifffile
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib
matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt


def plot_slice(data, slice_idx, pixel_size,
               cmap='grey',
               title=None,
               vmin=None,
               vmax=None,
               type='phase',
               figsize=(10, 8),
               fontsize=16):
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
    if type == 'phase':
        cbar.set_label(r'Electron density $\rho_e$ [1/nm$^3$]', size=fontsize, labelpad=15)
    else:
        cbar.set_label(r'Attenuation coeff. $\mu$ [1/nm]', size=fontsize, labelpad=15)
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


path = '/data/mct/22203/results/K3_2E/recon_phase/recon_cs01000_idx_0946.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

pmma_theory = 383.26
pmma_measured = 440.94
calibration = pmma_theory / pmma_measured
print('Calbration factor:', calibration)
file *= calibration

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=287 * calibration,
                     vmax=380 * calibration,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
#plt.savefig(os.path.join(home_dir, f'k3-1n_phase.png'), dpi=1200, bbox_inches='tight')
#plt.show()

path = '/data/mct/22203/results/K3_2E/recon_att/recon_cs01000_idx_0946.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=-1,
                     vmax=3.5,
                     type="att",
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
#plt.savefig(os.path.join(home_dir, f'k3-1n.png'), dpi=1200, bbox_inches='tight')
plt.show()

########################################################################################################################

path = '/data/mct/22203/results/K3_3H_Manual/recon_phase/recon_cs01000_idx_1259.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

pmma_theory = 383.26
pmma_measured = 490
calibration = pmma_theory / pmma_measured
print('Calbration factor:', calibration)
file *= calibration

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=287 * calibration,
                     vmax=450 * calibration,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
plt.show()
#plt.savefig(os.path.join(home_dir, f'k3-1n_phase.png'), dpi=1200, bbox_inches='tight')
#plt.show()

path = '/data/mct/22203/results/K3_3H_Manual/recon_att/recon_cs01000_idx_1259.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=-1,
                     vmax=9,
                     type="att",
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
#plt.savefig(os.path.join(home_dir, f'k3-1n.png'), dpi=1200, bbox_inches='tight')
plt.show()


########################################################################################################################


path = '/data/mct/22203/results/K3_1N/recon_phase/recon_cs01000_idx_0856.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

pmma_theory = 383.26
pmma_measured = 445
calibration = pmma_theory / pmma_measured
print('Calbration factor:', calibration)
file *= calibration

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=290 * calibration,
                     vmax=384 * calibration,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
plt.show()
#plt.savefig(os.path.join(home_dir, f'k3-1n_phase.png'), dpi=1200, bbox_inches='tight')
#plt.show()

path = '/data/mct/22203/results/K3_1N/recon_att/recon_cs01000_idx_0856.tiff'
file = tifffile.imread(path)
# Example usage:
# Your pixel size
binning_factor = 1
psize = 1.444e-6 * binning_factor  # 4x binned data

# Example with your data:
fig, ax = plot_slice(file,
                     slice_idx=0,
                     pixel_size=psize,
                     vmin=-1,
                     vmax=3.5,
                     type="att",
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
#plt.savefig(os.path.join(home_dir, f'k3-1n.png'), dpi=1200, bbox_inches='tight')
plt.show()