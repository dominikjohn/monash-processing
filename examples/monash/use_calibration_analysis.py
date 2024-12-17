import matplotlib
matplotlib.use('TkAgg', force=True)
from monash_processing.postprocessing.calibration_analysis import CalibrationAnalysis

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
}

# Initialize the calibration analysis
calibration = CalibrationAnalysis(materials, energy_keV=25)

# Set up analysis parameters
base_path = "/data/mct/22203/results/P6_Manual"
material_slices = [250, 250, 1070, 1300, 1900]

# Load the reconstruction stacks
phase_stack, att_stack = calibration.load_reconstruction_stacks(base_path, max_slices=2050)

# Perform the analysis
phase_results, att_results = calibration.analyze_materials(
    material_slices,
    n_slices=50,
    use_att=False,
    phase_correction_factor=0.32/0.155
)

# Plot the results
calibration.plot_phase_vs_attenuation(phase_results, att_results)