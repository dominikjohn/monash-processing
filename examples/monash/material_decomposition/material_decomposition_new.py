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

binning_factor = 1
psize = 1.444e-6 * binning_factor
energy = 25000
energy_keV = energy / 1000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
wavelength = 1.24e-9 / energy_keV
preview_slice = 971

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
rho_1 = 305.6
mu_1 = 0.576

rho_2 = CalibrationAnalysis.calculate_electron_density(lead['density'], lead['molecular_weight'], lead['electrons'])
mu_2 = CalibrationAnalysis.calculate_attenuation(lead['composition'], lead['density'], energy_keV)

matrix = np.array([[rho_1, rho_2],
                   [mu_1, mu_2]])

inverse = np.linalg.inv(matrix)

plot_slice(edensity_slice, slice_idx=0, pixel_size=psize, title="Electron density", vmin=230, vmax=330)

def plot_slice(data, slice_idx, pixel_size,
               cmap='grey',
               title=None,
               vmin=None,
               vmax=None,
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
    cbar.set_label(f'{title}', size=fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    return fig, ax

fig, ax = plot_slice(n2_volume.transpose(1, 0, 2),
                     slice_idx=150,
                     pixel_size=psize,
                     vmin=0,
                     vmax=1.2,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
plt.savefig(os.path.join(home_dir, f'mat_decomp_pvc_8binned.png'), dpi=1200, bbox_inches='tight')
plt.show()

fig, ax = plot_slice(n1_volume.transpose(1, 0, 2),
                     slice_idx=150,
                     pixel_size=psize,
                     vmin=0,
                     vmax=1.8,
                     title="",
                     cmap='grey')
home_dir = os.path.expanduser('~')
plt.savefig(os.path.join(home_dir, f'mat_decomp_ethanol_8binned.png'), dpi=1200, bbox_inches='tight')
plt.show()

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