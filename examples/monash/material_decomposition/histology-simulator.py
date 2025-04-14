import scipy.constants
import numpy as np
from pathlib import Path
import os
from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis
import tifffile
import glob

binning_factor = 1
psize = 1.444e-6 * binning_factor
energy = 25000
energy_keV = energy / 1000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
wavelength = 1.24e-9 / energy_keV
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

lead = {
    'density': 11.35,
    'molecular_weight': 207.2,
    'electrons': 82,
    'composition': {'Pb': 1}
}

slices = []
edensity_slice = load_single_tiff('umpa_window1', 'recon_phase', binning_factor, preview_slice)
mu_slice = load_single_tiff('umpa_window1', 'recon_att', binning_factor, preview_slice)

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

n1_slice = inverse[0, 0] * edensity_slice + inverse[0, 1] * mu_slice
n2_slice = inverse[1, 0] * edensity_slice + inverse[1, 1] * mu_slice
#plot_slice(n2_slice, slice_idx=0, pixel_size=psize, title="Lead [v/v], %", vmin=0, vmax=2, percent=True)
#plt.show()
v_m = n2_slice
rho_m = lead['density']
M_m = 207.2 # Lead
c_m = v_m * rho_m / M_m

#plot_slice(c_m * 1000, slice_idx=0, pixel_size=psize, title="Lead concentration [mmol/L]", vmin=0.0, vmax=1)
#plt.show()

from monash_processing.postprocessing.colorize import Colorizer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#base_path = '/Users/dominikjohn/Library/Mobile Documents/com~apple~CloudDocs/Documents/1_Projects/Paper Material Decomposition/visiblelight'
base_path = '/user/home'
colorizer = Colorizer(base_path)
result_dict = colorizer.importer(base_path)

wavelengths = result_dict['wavelengths']
hematin = result_dict['haematoxylin']

plt.plot(wavelengths, hematin)
plt.ylabel('$\epsilon$ [1/(cm * M)]')
plt.show()

#colorizer.display_data(wavelengths, hematin, concentration=10e-4)
#color_hex = colorizer.concentration_to_color(wavelengths, hematin, concentration=c_m[800:-800, 800:-800], thickness_um=100)
color_hex = colorizer.concentration_to_color(wavelengths, hematin, concentration=c_m[850:1200, 600:900], thickness_um=3, light_color=5500)


plt.clf()  # Clear current figure
plt.figure()

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
    plt.tight_layout()
    plt.title("RGB image from hex array")

plt.show()
