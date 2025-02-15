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

# Maximum solubility of I2 in ethanol is 240 g/L = 24 w/v% = ~30.4 wt%
max_solubility_wv = 24  # w/v%
percentages = np.linspace(0, max_solubility_wv, 25)  # w/v%
mu_total = []

for p in percentages:
    # Convert w/v% to mass fractions
    mass_I2 = p  # grams per 100 mL
    volume_ethanol = 100  # mL
    mass_ethanol = volume_ethanol * rho_ethanol
    total_mass = mass_I2 + mass_ethanol

    w_I2 = mass_I2 / total_mass
    w_ethanol = 1 - w_I2

    # Calculate solution density (g/cm³)
    rho_mix = total_mass / volume_ethanol

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

# Calculate pure ethanol mu
w_C_pure = 24.022 / M_ethanol
w_H_pure = 6.048 / M_ethanol
w_O_pure = 15.999 / M_ethanol

mu_ethanol = (w_C_pure * xraylib.CS_Total(6, E) +
              w_H_pure * xraylib.CS_Total(1, E) +
              w_O_pure * xraylib.CS_Total(8, E)) * rho_ethanol

plt.figure(figsize=(10, 6))
plt.plot(percentages, mu_total)
plt.axhline(y=mu_ethanol, color='g', linestyle='--', label='Pure ethanol')
plt.xlabel('I₂ content (w/v%)')
plt.ylabel('Linear attenuation coefficient µ [1/cm]')
plt.title('Attenuation of I₂-ethanol mixture at 25 keV')
plt.legend()
plt.grid(True)
plt.show()

# Print values at specific concentrations
test_points = [1, 2.5, 5, 10, 24]  # including max solubility
mu_test_points = np.interp(test_points, percentages, mu_total)

print("\nAttenuation coefficients at specific concentrations:")
for conc, mu in zip(test_points, mu_test_points):
    print(f"At {conc:>4}% I₂: µ = {mu:.2f} cm⁻¹")

# Calculate relative differences
print("\nRelative differences:")
rel_diffs = np.diff(mu_test_points) / mu_test_points[:-1] * 100
rel_diffs_ethanol = mu_test_points / mu_ethanol * 100

for i, diff in enumerate(rel_diffs):
    print(f"Relative increase from {test_points[i]}% to {test_points[i + 1]}%: {diff:.1f}%")
    print(f"Relative to pure ethanol at {test_points[i]}%: {rel_diffs_ethanol[i]:.1f}%")