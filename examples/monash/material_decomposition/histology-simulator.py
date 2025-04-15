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
from monash_processing.postprocessing.colorize import ColourSystem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#base_path = '/Users/dominikjohn/Library/Mobile Documents/com~apple~CloudDocs/Documents/1_Projects/Paper Material Decomposition/visiblelight'
base_path = '/user/home'
colorizer = Colorizer(base_path)
result_dict = colorizer.import_absorbances()

wavelengths = result_dict['wavelengths']
hematin_absorbance = result_dict['absorbances']
hematin_epsilon = hematin_absorbance * 2.2e4

plt.plot(wavelengths, hematin_epsilon)
plt.ylabel('$\epsilon$ [1/(cm * M)]')
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Example 2D concentration array
concentrations_2d = c_m
thickness_um = 5

# Load CMF data
cmf_path = os.path.join(colorizer.base_path, 'cie-cmf.txt')
cmf_data = np.loadtxt(cmf_path)

output = colorizer.calculate_transmitted_spectrum(
            wavelengths, hematin_epsilon,
            thickness_um=thickness_um,
            concentration=.5e-3,
            light_color=6500
        )

wavelengths = result_dict['wavelengths']
transmitted_spectrum = output['transmitted_spectrum'][0]
plt.plot(wavelengths, output['transmitted_spectrum'][0])
plt.plot(wavelengths, output['source_spectrum'])
plt.show()

x_bar = np.interp(wavelengths, cmf_data[:, 0], cmf_data[:, 1])
y_bar = np.interp(wavelengths, cmf_data[:, 0], cmf_data[:, 2])
z_bar = np.interp(wavelengths, cmf_data[:, 0], cmf_data[:, 3])

# Compute cone (XYZ) activations via integration
X = np.trapz(transmitted_spectrum * x_bar, wavelengths)
Y = np.trapz(transmitted_spectrum * y_bar, wavelengths)
Z = np.trapz(transmitted_spectrum * z_bar, wavelengths)

cone_labels = ['X', 'Y', 'Z']
cone_values = [X, Y, Z]

plt.bar(cone_labels, cone_values, color=['#555555', '#777777', '#999999'])
plt.ylabel("Activation")
plt.title("XYZ activation")
plt.tight_layout()
plt.show()

# Source: https://stackoverflow.com/questions/66360637/which-matrix-is-correct-to-map-xyz-to-linear-rgb-for-srgb
M_XYZ_to_RGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])

XYZ = np.array([X, Y, Z])
RGB = M_XYZ_to_RGB @ XYZ

plt.figure()
plt.bar(cone_labels, RGB, color=['red', 'green', 'blue'])
plt.ylabel("Activation")
plt.title("RGB")
plt.show()

#concentrations_2d = np.clip(c_m, 0, None)

#concentrations_2d = np.tile(np.array([0, 0.0001, 0.001, 0.005]), (50, 50))
concentrations_2d = c_m

output = colorizer.calculate_transmitted_spectrum(
            wavelengths, hematin_epsilon,
            thickness_um=1000,
            concentration=concentrations_2d,
            light_color=6500
        )

spectra = output['transmitted_spectrum'].reshape(*concentrations_2d.shape, len(wavelengths))
X = np.trapz(spectra * x_bar, wavelengths)
Y = np.trapz(spectra * y_bar, wavelengths)
Z = np.trapz(spectra * z_bar, wavelengths)

original_shape = concentrations_2d.shape

XYZ_values = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)  # Shape: (100, 3)
# Perform the matrix multiplication for all values at once
RGB_flat = XYZ_values @ M_XYZ_to_RGB.T  # Shape: (100, 3)
# Reshape back to original 10×10 grid

RGB = RGB_flat.reshape(*original_shape, 3).astype(int) / np.array([98, 93, 97])

def gamma_correct(rgb):
    rgb = np.clip(rgb, 0, 1)
    threshold = 0.0031308
    return np.where(
        rgb <= threshold,
        12.92 * rgb,
        1.055 * np.power(rgb, 1/2.4) - 0.055
    )

RGB_corrected = gamma_correct(RGB)
plt.imshow(RGB_corrected)
plt.show()

plt.figure(figsize=(10, 8))

# Background RGB image
#plt.imshow(RGB_corrected)

# Overlay n2_slice with a solid pink colormap
pink_rgb = np.array([200, 81, 204]) / 255

# Convert grayscale to RGB tinted pink
pink_overlay = np.clip(n1_slice, 0, 1)[..., np.newaxis] * pink_rgb

plt.imshow(pink_overlay, alpha=0.2)  # semi-transparent overlay

edensity_range = (0.8, 1)
edensity_norm = (n1_slice - edensity_range[0]) / (edensity_range[1] - edensity_range[0])
edensity_norm = np.clip(edensity_norm, 0, 1)

from matplotlib.colors import to_rgb
pink_color = to_rgb('#e7acc5')  # Pink for electron density
rgb_image = np.ones((*n1_slice.shape, 3))
for i in range(3):
    # Linear interpolation from white to pink based on normalized intensity
    rgb_image[..., i] = rgb_image[..., i] * (1 - edensity_norm) + pink_color[i] * edensity_norm

plt.imshow(RGB_corrected, alpha=1)
plt.imshow(rgb_image, alpha=.6)
plt.title("RGB with Eosin Overlay")
plt.axis("off")
plt.tight_layout()
plt.show()