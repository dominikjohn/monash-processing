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
#preview_slice = 971
preview_slice = 150
home_dir = os.path.expanduser('~')

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

def load_single_tiff(subfolder, channel, binning_factor=1, preview_slice=0) -> np.ndarray:
    tiff_path = Path("/data/mct/22203/results/K3_3H_ReverseOrder") / subfolder / channel
    if binning_factor != 1:
        tiff_path = tiff_path / f'binned{binning_factor}'

    print(str(tiff_path) + f'/*{preview_slice}.tif*')
    tiff_file = glob.glob(str(tiff_path) + f'/*{str(preview_slice).zfill(4)}.tiff')[0]
    data = tifffile.imread(tiff_file)
    return np.array(data)

lead = {
    'density': 11.35,
    'molecular_weight': 207.2,
    'electrons': 82,
    'composition': {'Pb': 1}
}

slices = []
for i in range(10):
    edensity_slice = load_single_tiff('umpa_window1', 'recon_phase', binning_factor, preview_slice+i)
    mu_slice = load_single_tiff('umpa_window1', 'recon_att', binning_factor, preview_slice+i)

    calibration = 383./495.
    edensity_slice *= calibration
    # soft tissue as determined by native measurement (K3.1N)
    rho_1 = 300
    mu_1 = 0.576

    rho_2 = CalibrationAnalysis.calculate_electron_density(lead['density'], lead['molecular_weight'], lead['electrons'])
    mu_2 = CalibrationAnalysis.calculate_attenuation(lead['composition'], lead['density'], energy_keV)

    matrix = np.array([[rho_1, rho_2],
                       [mu_1, mu_2]])
    inverse = np.linalg.inv(matrix)

    #plot_slice(edensity_slice, slice_idx=0, pixel_size=psize, title="Electron density [1/nm$^3$]", vmin=230, vmax=350, fontsize=16*1.75, colorbar_position='left')
    #plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_edensity.png'), dpi=900, bbox_inches='tight')
    #plt.show()

    #plot_slice(mu_slice, slice_idx=0, pixel_size=psize, title="Attenuation coefficient [1/m]", vmin=0, vmax=12, fontsize=16*1.75)
    #plt.savefig(os.path.join(home_dir, f'k3_3h_reverse_attenuation.png'), dpi=900, bbox_inches='tight')
    #plt.show()
    n1_slice = inverse[0, 0] * edensity_slice + inverse[0, 1] * mu_slice
    n2_slice = inverse[1, 0] * edensity_slice + inverse[1, 1] * mu_slice
    #plot_slice(n2_slice, slice_idx=0, pixel_size=psize, title="Lead [v/v], %", vmin=0, vmax=2, percent=True)
    #plt.show()
    v_m = n2_slice
    rho_m = lead['density']
    M_m = 207.2 # Lead
    c_m = v_m * rho_m / M_m

    slices.append(c_m)

c_m = np.mean(slices, axis=0) * 10
plot_slice(c_m, slice_idx=0, pixel_size=psize, title="Lead concentration [mol/l]")
plt.show()



from monash_processing.postprocessing.colorize import Colorizer
#base_path = '/Users/dominikjohn/Library/Mobile Documents/com~apple~CloudDocs/Documents/1_Projects/Paper Material Decomposition/visiblelight'
base_path = '/user/home'
colorizer = Colorizer()
result_dict = colorizer.importer(base_path)

wavelengths = result_dict['wavelengths']
hematin = result_dict['haematoxylin']

colorizer.display_data(wavelengths, hematin, concentration=10e-4)
raise Exception()

color_hex = colorizer.concentration_to_color(new_wavelengths, new_hematin, concentration=1e-3, thickness_um=200)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if isinstance(color_hex, str):
    rgb = mcolors.to_rgb(color_hex)  # Converts to (r, g, b) in [0, 1]
    img = np.ones((100, 100, 3)) * rgb
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{color_hex.upper()} → RGB: {tuple(np.round(rgb, 2))}")
elif isinstance(color_hex, np.ndarray) and color_hex.ndim == 2:
    # Assume 2D array of hex strings
    h, w = color_hex.shape
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            img[i, j] = mcolors.to_rgb(color_hex[i, j])
    plt.imshow(img)
    plt.axis('off')
    plt.title("RGB image from hex array")

plt.show()
