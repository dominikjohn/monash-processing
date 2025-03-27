import matplotlib

from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis
import matplotlib.pyplot as plt

# Define materials dictionary
materials = {
    'Ethanol': {
        'density': 0.789,
        'molecular_weight': 46.068,
        'electrons': 26,
        'composition': {'C': 2, 'H': 6, 'O': 1}
    },
    'PVC': {
        'density': 1.4,
        'molecular_weight': 62.5,
        'electrons': 32,
        'composition': {'C': 2, 'H': 3, 'Cl': 1}
    },
    'PTFE': {
        'density': 2.2,
        'molecular_weight': 100.02,
        'electrons': 48,
        'composition': {'C': 2, 'F': 4}
    },
    'POM': {
        'density': 1.41,
        'molecular_weight': 30.026,
        'electrons': 16,
        'composition': {'C': 1, 'H': 2, 'O': 1}
    },
    'PMMA': {
        'density': 1.18,
        'molecular_weight': 100.12,
        'electrons': 54,
        'composition': {'C': 5, 'H': 8, 'O': 2}
    },
    #'Lead': {
    #    'density': 11.35,
    #    'molecular_weight': 207.2,
    #    'electrons': 82,
    #    'composition': {'Pb': 1}
    #}
}

# Initialize the calibration analysis
calibration = CalibrationAnalysis(materials, energy_keV=25)
# Load the reconstruction stacks
#base_path = "/data/mct/22203/results/P6_ReverseOrder"
base_path = "/data/mct/22203/results/P6_Manual"

phase_stack, att_stack = calibration.load_reconstruction_stacks(base_path, max_slices=2050, bin_factor=4)

# Set up analysis parameters
# base_path = "/data/mct/22203/results/P6_Manual"
#base_path = "/data/mct/22203/results/P5_Manual"
material_slices = [1200, 550, 1070, 1505, 1850]
#material_slices = [250, 250, 1070, 1300, 1900]

def bin_z_direction(stack, z_bin_factor):
    # Get the new z dimension size
    new_z = stack.shape[0] // z_bin_factor

    # Reshape to prepare for z-binning and take mean
    # Assuming stack shape is (z, y, x)
    reshaped = stack[:new_z * z_bin_factor]  # Trim any extra slices
    binned = reshaped.reshape(new_z, z_bin_factor, *stack.shape[1:]).mean(axis=1)

    return binned

# Apply to both stacks
z_bin_factor = 4  # Or whatever factor you want to use
phase_stack_binned = bin_z_direction(phase_stack, z_bin_factor)
att_stack_binned = bin_z_direction(att_stack, z_bin_factor)

# Initial correction factor
# initial_correction = 0.315/0.155
initial_correction = 1
print(f"\nInitial correction factor: {initial_correction:.4f}")

# Perform analysis
phase_results, att_results = calibration.analyze_materials(
    material_slices,
    n_slices=70,
    use_att=True,
    phase_correction_factor=initial_correction,
    overwrite=False
)

font_params = {
    'title_size': 14,  # Size of plot titles
    'label_size': 18,  # Size of axis labels
    'tick_size': 16,  # Size of tick labels
    'legend_size': 16  # Size of legend text (for phase vs attenuation plot)
}

# Plot initial results
calibration.plot_phase_vs_attenuation(phase_results, att_results, font_params=font_params)

# Calculate theoretical electron density for PMMA
pmma_density = materials['PMMA']['density']
pmma_mw = materials['PMMA']['molecular_weight']
pmma_electrons = materials['PMMA']['electrons']
theoretical_ed = calibration.calculate_electron_density(pmma_density, pmma_mw, pmma_electrons)

# Get measured phase signal for PMMA (assuming same order as in materials dictionary)
pmma_idx = list(materials.keys()).index('PMMA')
measured_phase = phase_results[pmma_idx][0]

# Calculate new correction factor
new_correction = initial_correction * (theoretical_ed / measured_phase)
print(f"\nBased on PMMA:")
print(f"Theoretical electron density: {theoretical_ed:.2f}")
print(f"Measured phase signal: {measured_phase:.2f}")
print(f"New correction factor: {new_correction:.4f}")

# Run analysis with new correction
phase_results_corrected, att_results_corrected = calibration.analyze_materials(
    material_slices,
    n_slices=70,
    use_att=True,
    phase_correction_factor=new_correction
)

# Plot corrected results
calibration.plot_phase_vs_attenuation(phase_results_corrected, att_results_corrected, font_params=font_params)
import numpy as np
import os

# Create a dictionary to store all the important values
export_data = {
    # Material properties
    'materials': list(materials.keys()),

    # Theoretical values
    'theoretical_electron_densities': [],
    'theoretical_attenuations': [],

    # Original measurement results
    'phase_means': [res[0] for res in phase_results if res is not None],
    'phase_stds': [res[1] for res in phase_results if res is not None],
    'att_means': [res[0] for res in att_results if res is not None],
    'att_stds': [res[1] for res in att_results if res is not None],

    # Corrected measurement results
    'phase_means_corrected': [res[0] for res in phase_results_corrected if res is not None],
    'phase_stds_corrected': [res[1] for res in phase_results_corrected if res is not None],
    'att_means_corrected': [res[0] for res in att_results_corrected if res is not None],
    'att_stds_corrected': [res[1] for res in att_results_corrected if res is not None],

    # Correction factors
    'initial_correction_factor': initial_correction,
    'new_correction_factor': new_correction
}

# Get theoretical values to add to the dictionary
electron_densities, attenuations = calibration.calculate_theoretical_values()
for material in materials:
    export_data['theoretical_electron_densities'].append(electron_densities[material])
    export_data['theoretical_attenuations'].append(attenuations[material])

# Define the output path
output_path = os.path.join('/user/home/Desktop', 'calibration_results_p6manual.npz')

# Save the data using numpy's compressed format
np.savez(output_path, **export_data)

print("\nData exported successfully to: {}".format(output_path))

# If you want to also save a plain text summary file
summary_path = os.path.join('/user/home/Desktop', 'calibration_summary_p6manual.txt')
with open(summary_path, 'w') as f:
    f.write("Calibration Analysis Summary\n")
    f.write("===========================\n\n")

    f.write("Material Properties:\n")
    f.write("-----------------\n")
    for mat, props in materials.items():
        f.write("{}: density={} g/cm^3, MW={} g/mol, electrons={}\n".format(
            mat, props['density'], props['molecular_weight'], props['electrons']))

    f.write("\nTheoretical Values:\n")
    f.write("-----------------\n")
    for i, mat in enumerate(materials.keys()):
        f.write("{}: electron density={:.2f} e-/nm^-3, attenuation={:.4f} cm^-1\n".format(
            mat, electron_densities[mat], attenuations[mat]))

    f.write("\nMeasured Values (Initial):\n")
    f.write("------------------------\n")
    for i, mat in enumerate(materials.keys()):
        if i < len(export_data['phase_means']):
            f.write("{}: phase={:.2f}±{:.2f}, attenuation={:.4f}±{:.4f}\n".format(
                mat, export_data['phase_means'][i], export_data['phase_stds'][i],
                export_data['att_means'][i], export_data['att_stds'][i]))

    f.write("\nMeasured Values (Corrected):\n")
    f.write("---------------------------\n")
    for i, mat in enumerate(materials.keys()):
        if i < len(export_data['phase_means_corrected']):
            f.write("{}: phase={:.2f}±{:.2f}, attenuation={:.4f}±{:.4f}\n".format(
                mat, export_data['phase_means_corrected'][i], export_data['phase_stds_corrected'][i],
                export_data['att_means_corrected'][i], export_data['att_stds_corrected'][i]))

    f.write("\nCorrection Factors:\n")
    f.write("Initial: {:.4f}\n".format(initial_correction))
    f.write("Final: {:.4f}\n".format(new_correction))

print("Summary text exported to: {}".format(summary_path))
