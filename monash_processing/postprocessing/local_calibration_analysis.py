import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the saved numpy file
data_path = '/files/calibration_results.npz'

# Load the saved data
print(f"Loading data from {data_path}")
data = np.load(data_path)

# Extract the data
materials = data['materials']
theoretical_electron_densities = data['theoretical_electron_densities']
theoretical_attenuations = data['theoretical_attenuations']

# Original measurements
phase_means = data['phase_means']
phase_stds = data['phase_stds']
att_means = data['att_means']
att_stds = data['att_stds']

# Corrected measurements (if you want to use these instead)
phase_means_corrected = data['phase_means_corrected']
phase_stds_corrected = data['phase_stds_corrected']
att_means_corrected = data['att_means_corrected']
att_stds_corrected = data['att_stds_corrected']

# Choose which set of data to plot (original or corrected)
# Set this to True to use the corrected data
use_corrected = True

if use_corrected:
    plot_phase_means = phase_means_corrected
    plot_phase_stds = phase_stds_corrected
    plot_att_means = att_means_corrected
    plot_att_stds = att_stds_corrected
    title_suffix = "(Corrected)"
else:
    plot_phase_means = phase_means
    plot_phase_stds = phase_stds
    plot_att_means = att_means
    plot_att_stds = att_stds
    title_suffix = "(Original)"

# Font parameters
font_params = {
    'title_size': 14,
    'label_size': 18,
    'tick_size': 16,
    'legend_size': 14  # Smaller legend size
}

# Create figure with specified size
plt.figure(figsize=(10, 8))

# Define plot styles
colors = ['blue', 'red', 'green', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'v']

# Create empty lists for legend handles and labels
legend_handles = []
legend_labels = []

# Plot both theoretical and measured data for each material
for i, material in enumerate(materials):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]

    # Plot measured points with error bars
    measured_handle = plt.errorbar(
        plot_phase_means[i], plot_att_means[i],
        xerr=plot_phase_stds[i], yerr=plot_att_stds[i],
        color=color, marker=marker, markersize=8,
        capsize=5, capthick=1, linestyle='-',
        label=f"{material} (measured)"
    )

    # Plot theoretical points (hollow marker)
    theoretical_handle = plt.scatter(
        theoretical_electron_densities[i], theoretical_attenuations[i],
        color=color, marker=marker, s=100,
        facecolors='none', edgecolors=color, linewidth=2,
        label=f"{material} (theoretical)"
    )

    # Add a single entry to the legend for this material
    from matplotlib.lines import Line2D

    custom_handle = Line2D([0], [0], color=color, marker=marker,
                           markersize=8, label=material)
    legend_handles.append(custom_handle)
    legend_labels.append(f"{material} (○: theoretical, ┼: measured)")

# Set labels and other plot features
plt.xlabel('Electron density [1/nm³]', fontsize=font_params['label_size'])
plt.ylabel('Attenuation coefficient $\mu$ [1/cm]', fontsize=font_params['label_size'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.title(f'Phase vs Attenuation {title_suffix}', fontsize=font_params['title_size'])

# Add the custom legend with combined entries
plt.legend(handles=legend_handles, labels=legend_labels, fontsize=font_params['legend_size'])
plt.tick_params(axis='both', labelsize=font_params['tick_size'])

# Display the plot
plt.tight_layout()
plt.savefig('/home/user/calibration_plot.png', dpi=300)
plt.show()

# Print summary of the data
print("\nMaterial Summary:")
print("-" * 70)
print(f"{'Material':<10} {'Electron Density':<20} {'Theoretical μ':<15} {'Measured μ':<15}")
print(f"{'       ':<10} {'(electrons/nm³)':<20} {'(cm⁻¹)':<15} {'(cm⁻¹)':<15}")
print("-" * 70)
for i, material in enumerate(materials):
    print(
        f"{material:<10} {theoretical_electron_densities[i]:>8.2f}{' ' * 11} {theoretical_attenuations[i]:>8.4f}{' ' * 6} {plot_att_means[i]:>8.4f}±{plot_att_stds[i]:.4f}")

# Optional: Export cleaned data to CSV for easy use in other applications
csv_path = '/home/user/calibration_data.csv'
with open(csv_path, 'w') as f:
    f.write(
        "Material,Electron_Density,Theoretical_Attenuation,Measured_Phase,Phase_StdDev,Measured_Attenuation,Attenuation_StdDev\n")
    for i, material in enumerate(materials):
        f.write(f"{material},{theoretical_electron_densities[i]:.4f},{theoretical_attenuations[i]:.6f},")
        f.write(f"{plot_phase_means[i]:.6f},{plot_phase_stds[i]:.6f},{plot_att_means[i]:.6f},{plot_att_stds[i]:.6f}\n")

print(f"\nData also exported as CSV to: {csv_path}")