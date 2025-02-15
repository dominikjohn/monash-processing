from monash_processing.core.data_loader import DataLoader
import scipy
import scipy.constants
import numpy as np
from pathlib import Path
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2
from skimage.measure import block_reduce
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis

matplotlib.use('TkAgg', force=True)  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt

binning_factor = 4
psize = 1.444e-6 * binning_factor

energy = 25000
energy_keV = energy / 1000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)

loader = DataLoader(Path("/data/mct/22203/"), "P6_Manual")
edensity_volume = loader.load_reconstruction('recon_phase', binning_factor=4)
mu_volume = loader.load_reconstruction('recon_att', binning_factor=4)

#edensity_volume = block_reduce(edensity_volume, (binning_factor, 1, 1), np.mean)
#mu_volume = block_reduce(mu_volume, (binning_factor, 1, 1), np.mean)

edensity_volume = block_reduce(edensity_volume, (binning_factor, 1, 1), np.mean)
mu_volume = block_reduce(mu_volume, (binning_factor*1, 1, 1), np.mean)
#filtered_mu_volume = cv2.medianBlur(mu_volume.astype(np.float32), 3)

calibration = .8567
edensity_volume *= calibration

m3_to_nm3 = 1e27
#delta = 2 * np.pi * edensity_volume * scipy.constants.physical_constants['classical electron radius'][0] * m3_to_nm3 / (
#        wavevec ** 2)
#beta = mu_volume / 2 * wavevec

# Pure ethanol
#material1 = {
#    'density': 0.789,
#    'molecular_weight': 46.068,
#    'electrons': 26,
#    'composition': {'C': 2, 'H': 6, 'O': 1}
#}

# 96 % ethanol mixture
material1 = {
    'density': 0.797,
    'molecular_weight': 44.661,
    'electrons': 25.2,
    'composition': {'C': 1.8996789727126806, 'H': 5.799357945425361, 'O': 1.0}
}

# Pure ethanol
#material1 = {
#    'density': 0.789,  # g/cm³
#    'molecular_weight': 46.068,  # g/mol
#    'electrons': 26,
#    'composition': {'C': 2, 'H': 6, 'O': 1}
#}

# PMMA
material2 = {
    'density': 1.18,
    'molecular_weight': 100.12,
    'electrons': 54,
    'composition': {'C': 5, 'H': 8, 'O': 2}
}

#PTFE
material2 = {
        'density': 2.2,
        'molecular_weight': 100.02,
        'electrons': 48,
        'composition': {'C': 2, 'F': 4}
    }

#PVC
material2 =  {
        'density': 1.4,
        'molecular_weight': 62.5,
        'electrons': 32,
        'composition': {'C': 2, 'H': 3, 'Cl': 1}
}

rho_1 = CalibrationAnalysis.calculate_electron_density(material1['density'],
                                                            material1['molecular_weight'],
                                                            material1['electrons'])
rho_2 = CalibrationAnalysis.calculate_electron_density(material2['density'],
                                                            material2['molecular_weight'],
                                                            material2['electrons'])

mu_1 = CalibrationAnalysis.calculate_attenuation(material1['composition'], material1['density'], energy_keV)
mu_2 = CalibrationAnalysis.calculate_attenuation(material2['composition'], material2['density'], energy_keV)

matrix = np.array([[rho_1, rho_2],
                   [mu_1, mu_2]])

inverse = np.linalg.inv(matrix)

n1_volume = inverse[0, 0] * edensity_volume + inverse[0, 1] * mu_volume
n2_volume = inverse[1, 0] * edensity_volume + inverse[1, 1] * mu_volume

preview_slice = 500

rho_values = np.clip(edensity_volume[preview_slice, :, :].ravel(), 300, 500)
mu_values = np.clip(mu_volume[preview_slice, :, :].ravel(), 0.5, 2)

n1_values = n1_volume[preview_slice, :, :].ravel()
n2_values = n2_volume[preview_slice, :, :].ravel()

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(rho_values, mu_values, s=1)
ax1.set_xlabel('Electron density')  # Changed from ax1.xlabel
ax1.set_ylabel('Attenuation')      # Changed from ax1.ylabel

ax2.scatter(n1_values, n2_values, s=1)
ax2.set_xlabel('Ethanol (v/v)')    # Changed from ax2.xlabel
ax2.set_ylabel('PMMA (v/v)')       # Changed from ax2.ylabel

plt.show()

imshow(n1_volume.transpose(1, 0, 2)[150], title='Ethanol (v/v)')
imshow(n2_volume.transpose(1, 0, 2)[280], title='PVC (v/v)')


def plot_slice(data, slice_idx, pixel_size,
               cmap='grey',
               title=None,
               vmin=None,
               vmax=None,
               figsize=(10, 8),
               fontsize=16,
               show_size=True):
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

    if show_size:
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
    cbar.set_label(f'{title} [v/v]', size=fontsize, labelpad=15)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    return fig, ax

fig, ax = plot_slice(n2_volume.transpose(1, 0, 2),
                     slice_idx=300,
                     pixel_size=psize,
                     vmin=0,
                     vmax=1.2,
                     title="PVC",
                     cmap='grey',
                     fontsize=24,
                     show_size=False,
                     )
home_dir = os.path.expanduser('~')
plt.savefig(os.path.join(home_dir, f'mat_decomp_pvc_4binned.png'), dpi=1200, bbox_inches='tight')
plt.show()


fig, ax = plot_slice(n1_volume.transpose(1, 0, 2),
                     slice_idx=300,
                     pixel_size=psize,
                     vmin=0,
                     vmax=2.6,
                     title="Ethanol",
                     cmap='grey',
                     fontsize=24,
                     show_size=False,
                     )

home_dir = os.path.expanduser('~')
plt.savefig(os.path.join(home_dir, f'mat_decomp_ethanol_4binned.png'), dpi=1200, bbox_inches='tight')
plt.show()
