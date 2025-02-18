import xraylib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Use default font if TUM font not available
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

rho_I2 = 4.93  # g/cm³ (solid iodine density)
rho_ethanol = 0.789  # g/cm³ at room temperature
E = 25  # keV

# Molecular masses
M_ethanol = 46.07  # g/mol (C2H5OH)
M_I2 = 253.81  # g/mol

# Using weight percentages directly
percentages = np.linspace(0, 30, 25)  # w/w%
mu_total = []

for p in percentages:
    # Direct weight fractions
    w_I2 = p / 100
    w_ethanol = 1 - w_I2

    # Calculate solution density (g/cm³) using simple mixing rule
    rho_mix = 1 / (w_I2 / rho_I2 + w_ethanol / rho_ethanol)

    # Mass fractions within ethanol portion (C2H5OH)
    w_C = w_ethanol * (24.022 / M_ethanol)  # 2 * 12.011
    w_H = w_ethanol * (6.048 / M_ethanol)  # 6 * 1.008
    w_O = w_ethanol * (15.999 / M_ethanol)

    # I2 mass fraction
    w_I = w_I2 * (2 * 126.90447 / M_I2)

    mu = (w_I * xraylib.CS_Total(53, E) +  # I
          w_C * xraylib.CS_Total(6, E) +  # C
          w_H * xraylib.CS_Total(1, E) +  # H
          w_O * xraylib.CS_Total(8, E)  # O
          ) * rho_mix

    mu_total.append(mu)


def calculate_electron_density(w_I, w_C, w_H, w_O, rho_mix):
    """
    Calculate electron density in electrons per nm³.
    """
    N_A = 6.02214076e23  # mol⁻¹
    rho_mix_nm = rho_mix * 1e-21

    e_density = N_A * rho_mix_nm * (
            w_I * (53 / 126.90447) +  # Z/M for I
            w_C * (6 / 12.0107) +  # Z/M for C
            w_H * (1 / 1.00784) +  # Z/M for H
            w_O * (8 / 15.999)  # Z/M for O
    )
    return e_density


# Calculate electron densities
e_densities = []
for p in percentages:
    w_I2 = p / 100
    w_ethanol = 1 - w_I2

    rho_mix = 1 / (w_I2 / rho_I2 + w_ethanol / rho_ethanol)

    w_C = w_ethanol * (24.022 / M_ethanol)
    w_H = w_ethanol * (6.048 / M_ethanol)
    w_O = w_ethanol * (15.999 / M_ethanol)
    w_I = w_I2 * (2 * 126.90447 / M_I2)

    e_density = calculate_electron_density(w_I, w_C, w_H, w_O, rho_mix)
    e_densities.append(e_density)

# Calculate pure ethanol values
w_C_pure = 24.022 / M_ethanol
w_H_pure = 6.048 / M_ethanol
w_O_pure = 15.999 / M_ethanol

mu_ethanol = (w_C_pure * xraylib.CS_Total(6, E) +
              w_H_pure * xraylib.CS_Total(1, E) +
              w_O_pure * xraylib.CS_Total(8, E)) * rho_ethanol

e_density_ethanol = calculate_electron_density(0, w_C_pure, w_H_pure, w_O_pure, rho_ethanol)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Attenuation plot
ax1.plot(percentages, mu_total)
ax1.axhline(y=mu_ethanol, color='g', linestyle='--', label='Pure ethanol')
ax1.set_xlabel('I₂ content (wt%)')
ax1.set_ylabel('Linear attenuation coefficient µ [1/cm]')
ax1.set_title('Attenuation of I₂-ethanol mixture at 25 keV')
ax1.legend()
ax1.grid(True)

# Electron density plot
ax2.plot(percentages, e_densities)
ax2.axhline(y=e_density_ethanol, color='g', linestyle='--', label='Pure ethanol')
ax2.set_xlabel('I₂ content (wt%)')
ax2.set_ylabel('Electron density [e⁻/nm³]')
ax2.set_title('Electron density of I₂-ethanol mixture')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print values at specific concentrations
test_points = [1, 2.5, 5, 10, 30]
mu_test_points = np.interp(test_points, percentages, mu_total)
e_density_test_points = np.interp(test_points, percentages, e_densities)

print("\nResults at test points:")
print("=" * 50)
for conc, mu, ed in zip(test_points, mu_test_points, e_density_test_points):
    print(f"\nAt {conc:>4}% I₂:")
    print(f"µ = {mu:.2f} cm⁻¹")
    print(f"Electron density = {ed:.2f} e⁻/nm³")
    print(f"Relative to ethanol (µ): {(mu / mu_ethanol - 1) * 100:.1f}%")
    print(f"Relative to ethanol (e⁻): {(ed / e_density_ethanol - 1) * 100:.1f}%")

# Calculate relative differences between consecutive points
rel_diffs_mu = np.diff(mu_test_points) / mu_test_points[:-1] * 100
print("\nRelative increases in attenuation:")
print("=" * 50)
for i, diff in enumerate(rel_diffs_mu):
    print(f"From {test_points[i]}% to {test_points[i + 1]}%: {diff:.1f}%")