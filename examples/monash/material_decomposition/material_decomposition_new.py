from monash_processing.core.data_loader import DataLoader
import scipy.constants
import numpy as np
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import glob


def plot_slice(data, slice_idx, pixel_size,
               cmap='grey',
               title=None,
               vmin=None,
               vmax=None,
               figsize=(10, 8),
               fontsize=16,
               percent=False,
               colorbar_position='right'):  # New parameter for colorbar position

    # Set the font size globally
    plt.rcParams.update({'font.size': fontsize})

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get the slice
    slice_data = data[slice_idx] if len(data.shape) == 3 else data

    if percent:
        # Plot the image
        im = ax.imshow(slice_data * 100,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax)
    else:
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

    # Position the colorbar according to the parameter
    if colorbar_position.lower() == 'left':
        cax = divider.append_axes("left", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        # For left position, we need to adjust the orientation of ticks
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    else:  # Default to right
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)

    cbar.set_label(f'{title}', size=fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    return fig, ax

binning_factor = 1
psize = 1.444e-6 * binning_factor
energy = 25000
energy_keV = energy / 1000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
wavelength = 1.24e-9 / energy_keV
#preview_slice = 971
preview_slice = 150
home_dir = os.path.expanduser('~')

def load_single_tiff(subfolder, channel, binning_factor=1, preview_slice=0) -> np.ndarray:
    tiff_path = Path("/data/mct/22203/results/K3_3H_ReverseOrder") / subfolder / channel
    if binning_factor != 1:
        tiff_path = tiff_path / f'binned{binning_factor}'

    print(str(tiff_path) + f'/*{preview_slice}.tif*')
    tiff_file = glob.glob(str(tiff_path) + f'/*{str(preview_slice).zfill(4)}.tiff')[0]
    data = tifffile.imread(tiff_file)
    return np.array(data)

edensity_slice = load_single_tiff('umpa_window1', 'recon_phase', binning_factor, preview_slice)
mu_slice = load_single_tiff('umpa_window1', 'recon_att', binning_factor, preview_slice)

calibration = 383./495.
print(f'Calibration: {calibration}')
edensity_slice *= calibration

lead = {
    'density': 11.35,
    'molecular_weight': 207.2,
    'electrons': 82,
    'composition': {'Pb': 1}
}

# soft tissue as determined by native measurement (K3.1N)
rho_1 = 300
mu_1 = 0.576

rho_2 = CalibrationAnalysis.calculate_electron_density(lead['density'], lead['molecular_weight'], lead['electrons'])
mu_2 = CalibrationAnalysis.calculate_attenuation(lead['composition'], lead['density'], energy_keV)

matrix = np.array([[rho_1, rho_2],
                   [mu_1, mu_2]])

inverse = np.linalg.inv(matrix)

plot_slice(edensity_slice, slice_idx=0, pixel_size=psize, title="Electron density [1/nm$^3$]", vmin=230, vmax=350, fontsize=16*1.75, colorbar_position='left')
#plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_edensity.png'), dpi=900, bbox_inches='tight')
plt.show()

plot_slice(mu_slice, slice_idx=0, pixel_size=psize, title="Attenuation coefficient [1/m]", vmin=0, vmax=12, fontsize=16*1.75)
plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_attenuation.png'), dpi=900, bbox_inches='tight')
plt.show()

n1_slice = inverse[0, 0] * edensity_slice + inverse[0, 1] * mu_slice
n2_slice = inverse[1, 0] * edensity_slice + inverse[1, 1] * mu_slice

plot_slice(n2_slice, slice_idx=0, pixel_size=psize, title="Lead [v/v], %", vmin=0, vmax=2, percent=True)
plt.show()

plot_slice(n1_slice, slice_idx=0, pixel_size=psize, title="Soft tissue [v/v], %", vmin=80, vmax=120, percent=True)
plt.show()


def plot_he_overlay(edensity_data, lead_fraction_data, pixel_size,
                    edensity_range=(230, 350), lead_range=(0, 0.01),
                    figsize=(12, 10), fontsize=16):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.colors as colors
    import numpy as np
    from matplotlib.colors import to_rgb

    plt.rcParams.update({'font.size': fontsize})

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get exact colors from hex codes
    purple_color = to_rgb('#480f62')  # Purple for lead
    pink_color = to_rgb('#e7acc5')  # Pink for electron density

    # Normalize the data using the provided range
    edensity_norm = (edensity_data - edensity_range[0]) / (edensity_range[1] - edensity_range[0])
    edensity_norm = np.clip(edensity_norm, 0, 1)

    # Normalize lead data
    lead_norm = lead_fraction_data / lead_range[1]
    lead_norm = np.clip(lead_norm, 0, 1)

    # Create an RGB image with white background
    rgb_image = np.ones((*edensity_data.shape, 3))

    # Apply electron density (pink) where it's present - transparent to full color
    for i in range(3):
        # Linear interpolation from white to pink based on normalized intensity
        rgb_image[..., i] = rgb_image[..., i] * (1 - edensity_norm) + pink_color[i] * edensity_norm

    # Apply lead (purple) with alpha blending - transparent to full color
    for i in range(3):
        # Blend the current colors with purple based on lead concentration
        rgb_image[..., i] = rgb_image[..., i] * (1 - lead_norm) + purple_color[i] * lead_norm

    # Display the RGB image
    im = ax.imshow(rgb_image)

    # Add scale bar
    scalebar = ScaleBar(pixel_size, "m", length_fraction=.2,
                        color='black', box_alpha=0,
                        location='lower center',
                        font_properties={'size': fontsize})
    ax.add_artist(scalebar)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Create custom colorbars for each component
    divider = make_axes_locatable(ax)

    # Left colorbar for electron density (pink)
    cax_left = divider.append_axes("left", size="5%", pad=0.5)
    pink_cmap = colors.LinearSegmentedColormap.from_list("pink",
                                                         [(1, 1, 1), pink_color])
    pink_norm = colors.Normalize(vmin=edensity_range[0], vmax=edensity_range[1])
    pink_sm = plt.cm.ScalarMappable(norm=pink_norm, cmap=pink_cmap)
    cbar_pink = plt.colorbar(pink_sm, cax=cax_left)
    cbar_pink.set_label('Electron density [1/nm$^3$]', size=fontsize, labelpad=15)
    cbar_pink.ax.tick_params(labelsize=fontsize)
    cbar_pink.ax.yaxis.set_ticks_position('left')
    cbar_pink.ax.yaxis.set_label_position('left')

    # Right colorbar for lead (purple)
    cax_right = divider.append_axes("right", size="5%", pad=0.15)
    purple_cmap = colors.LinearSegmentedColormap.from_list("purple",
                                                           [(1, 1, 1), purple_color])
    purple_norm = colors.Normalize(vmin=lead_range[0] * 100, vmax=lead_range[1] * 100)
    purple_sm = plt.cm.ScalarMappable(norm=purple_norm, cmap=purple_cmap)
    cbar_purple = plt.colorbar(purple_sm, cax=cax_right)
    cbar_purple.set_label('Lead [v/v] %', size=fontsize, labelpad=15)
    cbar_purple.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax

slicing = np.s_[850:-1450, 1000:-600]
fig, ax = plot_he_overlay(edensity_slice[slicing], n2_slice[slicing], psize,
                          edensity_range=(230, 310), lead_range=(0, 0.01))
#plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_virtual_histology.png'), dpi=900, bbox_inches='tight')
plt.show()
# Example usage:
# fig, ax = plot_overlay(edensity_slice, n2_slice, psize)
# plt.show()

# n1_slice = inverse[0, 0] * edensity_slice + inverse[0, 1] * mu_slice
# n2_slice = inverse[1, 0] * edensity_slice + inverse[1, 1] * mu_slice
#
# rho_values = np.clip(edensity_volume[slice, :, :].ravel(), 300, 500)
# mu_values = np.clip(mu_volume[slice, :, :].ravel(), 0.5, 2)
#
# n1_values = n1_volume[slice, :, :].ravel()
# n2_values = n2_volume[slice, :, :].ravel()
#
# plt.figure()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# ax1.scatter(rho_values, mu_values, s=1)
# ax1.set_xlabel('Electron density')  # Changed from ax1.xlabel
# ax1.set_ylabel('Attenuation')      # Changed from ax1.ylabel
#
# ax2.scatter(n1_values, n2_values, s=1)
# ax2.set_xlabel('Ethanol (v/v)')    # Changed from ax2.xlabel
# ax2.set_ylabel('PMMA (v/v)')       # Changed from ax2.ylabel
#
# plt.show()
#
# imshow(n1_volume.transpose(1, 0, 2)[150], title='Ethanol (v/v)')
# imshow(n2_volume.transpose(1, 0, 2)[280], title='PVC (v/v)')

def plot_he_overlay(edensity_data, lead_fraction_data, pixel_size,
                    edensity_range=(230, 350), lead_range=(0, 0.01),
                    figsize=(12, 10), fontsize=16, location='lower left', color='black'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.colors as colors
    import numpy as np
    from matplotlib.colors import to_rgb

    plt.rcParams.update({'font.size': fontsize})

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get exact colors from hex codes
    purple_color = to_rgb('#480f62')  # Purple for lead
    pink_color = to_rgb('#e7acc5')  # Pink for electron density

    # Normalize the data using the provided range
    edensity_norm = (edensity_data - edensity_range[0]) / (edensity_range[1] - edensity_range[0])
    edensity_norm = np.clip(edensity_norm, 0, 1)

    # Normalize lead data
    lead_norm = lead_fraction_data / lead_range[1]
    lead_norm = np.clip(lead_norm, 0, 1)

    rgb_image = np.ones((*edensity_data.shape, 3))

    # Apply electron density (pink) where it's present - transparent to full color
    for i in range(3):
        # Linear interpolation from white to pink based on normalized intensity
        rgb_image[..., i] = rgb_image[..., i] * (1 - edensity_norm) + pink_color[i] * edensity_norm

    # Apply lead (purple) with alpha blending - transparent to full color
    for i in range(3):
        # Blend the current colors with purple based on lead concentration
        rgb_image[..., i] = rgb_image[..., i] * (1 - lead_norm) + purple_color[i] * lead_norm

    im = ax.imshow(rgb_image)

    # Add scale bar with adjusted position
    # Using pad parameter to control position instead of offset
    scalebar = ScaleBar(pixel_size, "m", length_fraction=.25,
                        color=color, box_alpha=0,
                        location=location,  # Change to lower left
                        pad=0.1,
                        font_properties={'size': fontsize})
    ax.add_artist(scalebar)

    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)

    # Left colorbar for electron density (pink)
    cax_left = divider.append_axes("left", size="5%", pad=0.5)
    pink_cmap = colors.LinearSegmentedColormap.from_list("pink",
                                                         [(1, 1, 1), pink_color])
    pink_norm = colors.Normalize(vmin=edensity_range[0], vmax=edensity_range[1])
    pink_sm = plt.cm.ScalarMappable(norm=pink_norm, cmap=pink_cmap)
    cbar_pink = plt.colorbar(pink_sm, cax=cax_left)
    cbar_pink.set_label('Electron density [1/nm$^3$]', size=fontsize, labelpad=15)
    cbar_pink.ax.tick_params(labelsize=fontsize)
    cbar_pink.ax.yaxis.set_ticks_position('left')
    cbar_pink.ax.yaxis.set_label_position('left')

    # Right colorbar for lead (purple)
    cax_right = divider.append_axes("right", size="5%", pad=0.15)
    purple_cmap = colors.LinearSegmentedColormap.from_list("purple",
                                                           [(1, 1, 1), purple_color])
    purple_norm = colors.Normalize(vmin=lead_range[0] * 100, vmax=lead_range[1] * 100)
    purple_sm = plt.cm.ScalarMappable(norm=purple_norm, cmap=purple_cmap)
    cbar_purple = plt.colorbar(purple_sm, cax=cax_right)
    cbar_purple.set_label('Lead [v/v] %', size=fontsize, labelpad=15)
    cbar_purple.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    return fig, ax

slicing = np.s_[900:-1670, 900:-675]
fig, ax = plot_he_overlay(edensity_slice[slicing], n2_slice[slicing], psize,
                          edensity_range=(230, 310), lead_range=(0, 0.01), location='lower center', color='white')
plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_virtual_histology.png'), dpi=900, bbox_inches='tight')
plt.show()