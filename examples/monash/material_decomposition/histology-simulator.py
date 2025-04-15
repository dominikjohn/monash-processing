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
concentrations_2d = c_m[1000:1600, 800:1400]
thickness_um = 100

concentrations_2d = np.ones_like(concentrations_2d)[10:50, 10:50] * 0.1

# Load CMF data
cmf_path = os.path.join(colorizer.base_path, 'cie-cmf.txt')
cmf_data = np.loadtxt(cmf_path)

output = colorizer.calculate_transmitted_spectrum(
            wavelengths, hematin_epsilon,
            thickness_um=thickness_um,
            concentration=1,
            light_color=6500
        )

wavelengths = result_dict['wavelengths']
transmitted_spectrum = output['transmitted_spectrum'][0]
plt.plot(wavelengths, output['transmitted_spectrum'][0])
plt.plot(wavelengths, output['source_spectrum'])
plt.show()



def spectrum_to_rgb(wavelengths, spectrum, cmf_data, scale_brightness=True):
    interp_cmf = interp1d(cmf_data[:, 0], cmf_data[:, 1:], axis=0, bounds_error=False, fill_value=0)
    cmf_interp = interp_cmf(wavelengths)
    XYZ = np.trapz(spectrum[:, np.newaxis] * cmf_interp, wavelengths, axis=0)
    XYZ /= XYZ[1] + 1e-10  # normalize to luminance = 1
    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    rgb_linear = M @ XYZ
    rgb_linear = np.clip(rgb_linear, 0, 1)
    if scale_brightness:
        brightness = XYZ[1]
        rgb_linear *= brightness / (np.max(rgb_linear) + 1e-10)
    print("XYZ", XYZ)
    print("RGB", rgb_linear)
    return np.clip(rgb_linear, 0, 1)

# Convert 2D concentration map to RGB image
h, w = concentrations_2d.shape
img = np.zeros((h, w, 3))

for i in range(h):
    for j in range(w):
        c = concentrations_2d[i, j]
        output = colorizer.calculate_transmitted_spectrum(
            wavelengths, hematin_epsilon,
            thickness_um=thickness_um,
            concentration=c,
            light_color=6500
        )
        spectrum = output['transmitted_spectrum'][0]
        img[i, j] = spectrum_to_rgb(wavelengths, spectrum, cmf_data)

# Display image
plt.imshow(img)
plt.axis('off')
plt.title("Color map of concentration")
plt.show()