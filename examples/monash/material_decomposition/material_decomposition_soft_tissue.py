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
from scipy import fftpack

binning_factor = 4
psize = 1.444e-6 * binning_factor

energy = 25000
energy_keV = energy / 1000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
wavelength = 1.24e-9 / energy_keV
#loader = DataLoader(Path("/data/mct/22203/"), "P6_Manual")
loader = DataLoader(Path("/data/mct/22203/"), "K3_1N")
edensity_volume = loader.load_reconstruction('recon_phase', binning_factor=1)
mu_volume = loader.load_reconstruction('recon_att', binning_factor=1)

#calibration = .857
calibration = .87
edensity_volume *= calibration

#m3_to_nm3 = 1e27
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

lead = {
    'density': 11.35,
    'molecular_weight': 207.2,
    'electrons': 82,
    'composition': {'Pb': 1}
}

# soft tissue as determined by native measurement (K3.1N)
rho_1 = 305.6
mu_1 = 0.576

rho_2 = 1570
mu_2 = 340.69
#rho_2 = CalibrationAnalysis.calculate_electron_density(lead['density'], lead['molecular_weight'], lead['electrons'])
#mu_2 = CalibrationAnalysis.calculate_attenuation(lead['composition'], lead['density'], energy_keV)

matrix = np.array([[rho_1, rho_2],
                   [mu_1, mu_2]])

inverse = np.linalg.inv(matrix)

# Lead
delta = 2.9625e-06
beta = 2.083e-07
delta_beta_ratio = delta/beta

def paganin_filter(image, pixel_size, dist, wavelength, delta_beta_ratio):
    # Convert energy to wavelength
      # wavelength in meters

    # Get image dimensions
    ny, nx = image.shape

    # Create coordinate grids
    y, x = np.ogrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    y = fftpack.fftshift(y)
    x = fftpack.fftshift(x)

    # Calculate spatial frequencies
    kx = 2 * np.pi * x / (nx * pixel_size)
    ky = 2 * np.pi * y / (ny * pixel_size)
    k = np.sqrt(kx ** 2 + ky ** 2)

    # Create Paganin filter
    denom = 1 + wavelength * dist * delta_beta_ratio * k ** 2
    paganin_filter = 1 / denom

    # Apply filter in Fourier space
    image_fft = fftpack.fft2(image)
    filtered_fft = image_fft * paganin_filter
    filtered_image = np.real(fftpack.ifft2(filtered_fft))

    return filtered_image

edensity_slice = edensity_volume[1500]
mu_slice = mu_volume[1500]
filtered_mu_slice = paganin_filter(mu_slice, psize, 0.155, wavelength, 14.2)

n1_slice = inverse[0, 0] * edensity_slice + inverse[0, 1] * filtered_mu_slice
n2_slice = inverse[1, 0] * edensity_slice + inverse[1, 1] * filtered_mu_slice

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
    cbar.set_label(r'Ethanol [v/v]', size=fontsize, labelpad=15)
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



