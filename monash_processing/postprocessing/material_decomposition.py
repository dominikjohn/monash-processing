from monash_processing.core.data_loader import DataLoader
import scipy
import numpy as np
from pathlib import Path

energy = 25000
wavevec = 2 * np.pi * energy / (scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)

loader = DataLoader(Path("/data/mct/22203/"), "P6_Manual")
edensity_volume = loader.load_reconstruction('phase')
mu_volume = loader.load_reconstruction('attenuation')

delta = 2 * np.pi * edensity_volume * scipy.constants.physical_constants['electron radius'][0] / wavevec ** 2
beta = mu_volume / 2 * wavevec

material1 = {
    'Ethanol': {
        'density': 0.789,
        'molecular_weight': 46.068,
        'electrons': 26,
        'composition': {'C': 2, 'H': 6, 'O': 1}
    }
}

material2 = {
    'PMMA': {
        'density': 1.18,
        'molecular_weight': 100.12,
        'electrons': 54,
        'composition': {'C': 5, 'H': 8, 'O': 2}
    },
}
