from monash_processing.core.data_loader import DataLoader
import scipy
import scipy.constants
import numpy as np
from pathlib import Path
from monash_processing.utils.ImageViewer import ImageViewer as imshow
import cv2
import matplotlib

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

calibration = 384.01 / 314.33  # PMMA before correction vs after
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

# PMMA
material2 = {
    'density': 1.18,
    'molecular_weight': 100.12,
    'electrons': 54,
    'composition': {'C': 5, 'H': 8, 'O': 2}
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

rho_values = edensity_volume[preview_slice, :, :].ravel()
mu_values = mu_volume[preview_slice, :, :].ravel()

n1_values = n1_volume[preview_slice, :, :].ravel()
n2_values = n2_volume[preview_slice, :, :].ravel()

plt.figure()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.scatter(rho_values, mu_values, s=1)
ax1.xlabel('Electron density')
ax1.ylabel('Attenuation')

ax2.scatter(n1_values, n2_values, s=1)
ax2.xlabel('Ethanol (v/v)')
ax2.ylabel('PMMA (v/v)')
plt.show()


