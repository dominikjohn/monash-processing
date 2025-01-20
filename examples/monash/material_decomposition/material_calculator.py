from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis
import numpy as np

lead = {
    'density': 11.35,
    'molecular_weight': 207.2,
    'electrons': 82,
    'composition': {'Pb': 1}
}

lead = {
    'density': 1.18,
    'molecular_weight': 100.12,
    'electrons': 54,
    'composition': {'C': 5, 'H': 8, 'O': 2}
}
energy = 25
mu = CalibrationAnalysis.calculate_attenuation(lead['composition'], lead['density'], energy)
rho = CalibrationAnalysis.calculate_electron_density(lead['density'], lead['molecular_weight'], lead['electrons'])

print(f"Attenuation coefficient: {mu:.4f} 1/cm")
print(f"Electron density: {rho:.4f} electrons/nm^3")

print(f"Refractive index: {CalibrationAnalysis.electron_density_to_refractive_index(rho, 1):.4f}")

def electron_density_to_refractive_index(electron_density, wavelength):
    r_e = 2.818e-15  # classical electron radius in meters
    # Calculate delta (real part)
    return (r_e * wavelength ** 2 * electron_density) / (2 * np.pi)

import numpy as np

class MolecularCalculator:
    def __init__(self):
        import xraylib

        # Energy in keV
        self.energy = 25.0

        # Calculate mass attenuation coefficients at 25 keV using xraylib
        self.mass_attenuation = {
            'H': xraylib.CS_Total(1, self.energy),
            'C': xraylib.CS_Total(6, self.energy),
            'O': xraylib.CS_Total(8, self.energy),
            'Pb': xraylib.CS_Total(82, self.energy)
        }
        # Define atomic weights and electron counts
        self.atomic_data = {
            'H': {'weight': 1.008, 'electrons': 1},
            'C': {'weight': 12.011, 'electrons': 6},
            'O': {'weight': 15.999, 'electrons': 8},
            'Pb': {'weight': 207.2, 'electrons': 82}
        }

        # Hematein molecular formula: C16H14O6
        self.hematein = {
            'composition': {'C': 16, 'H': 14, 'O': 6},
            'density': 1.52  # g/cm³ (approximate)
        }

    def calculate_molecular_weight(self, composition):
        """Calculate molecular weight from composition dictionary"""
        return sum(count * self.atomic_data[element]['weight']
                   for element, count in composition.items())

    def calculate_total_electrons(self, composition):
        """Calculate total number of electrons from composition dictionary"""
        return sum(count * self.atomic_data[element]['electrons']
                   for element, count in composition.items())

    def calculate_electron_density(self, density, molecular_weight, total_electrons):
        """Calculate electron density in electrons/nm³"""
        # Convert g/cm³ to g/nm³
        density_nm3 = density * 1e-21

        # Calculate number of molecules per nm³
        molecules_per_nm3 = (density_nm3 / molecular_weight) * 6.022e23

        # Calculate electron density
        return molecules_per_nm3 * total_electrons

    def calculate_linear_attenuation(self, composition, density):
        """Calculate linear attenuation coefficient in 1/cm at 25 keV"""
        # Calculate mass attenuation coefficient
        mass_atten = sum(count * self.mass_attenuation[element]
                         for element, count in composition.items())

        # Convert to linear attenuation coefficient
        return mass_atten * density

    def calculate_complex_properties(self):
        # Create composition for Hematein-Pb complex
        complex_composition = self.hematein['composition'].copy()
        complex_composition['Pb'] = 1

        # Calculate molecular properties
        molecular_weight = self.calculate_molecular_weight(complex_composition)
        total_electrons = self.calculate_total_electrons(complex_composition)

        # Estimate density of complex (this is an approximation)
        # Using weighted average of hematein and lead densities
        hematein_weight = self.calculate_molecular_weight(self.hematein['composition'])
        complex_density = (self.hematein['density'] * hematein_weight +
                           11.35 * self.atomic_data['Pb']['weight']) / molecular_weight

        # Calculate electron density
        electron_density = self.calculate_electron_density(
            complex_density, molecular_weight, total_electrons)

        return {
            'molecular_weight': molecular_weight,
            'total_electrons': total_electrons,
            'complex_density': complex_density,
            'electron_density': electron_density,
            'complex_composition': complex_composition
        }

    def electron_density_to_refractive_index(self, electron_density, wavelength):
        """Calculate refractive index delta from electron density"""
        r_e = 2.818e-15  # classical electron radius in meters
        return (r_e * wavelength ** 2 * electron_density) / (2 * np.pi)


# Run calculations
calculator = MolecularCalculator()
results = calculator.calculate_complex_properties()
print(f"Molecular Weight: {results['molecular_weight']:.2f} g/mol")
print(f"Total Electrons: {results['total_electrons']}")
print(f"Complex Density: {results['complex_density']:.2f} g/cm³")
print(f"Electron Density: {results['electron_density']:.2e} electrons/nm³")

# Calculate linear attenuation coefficient
linear_atten = calculator.calculate_linear_attenuation(
    results['complex_composition'], results['complex_density'])
print(f"\nLinear Attenuation Coefficient at 25 keV: {linear_atten:.2f} 1/cm")