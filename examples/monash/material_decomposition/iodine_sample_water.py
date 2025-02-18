import xraylib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def calculate_electron_density(w_K, w_I, w_H, w_O, rho_mix):
    """
    Calculate electron density in electrons per nm³.

    Args:
        w_K: Mass fraction of Potassium
        w_I: Mass fraction of Iodine
        w_H: Mass fraction of Hydrogen
        w_O: Mass fraction of Oxygen
        rho_mix: Density of mixture in g/cm³

    Returns:
        Electron density in electrons/nm³
    """
    N_A = 6.02214076e23  # Avogadro's number
    rho_mix_nm = rho_mix * 1e-21  # Convert density to g/nm³

    e_density = N_A * rho_mix_nm * (
            w_K * (19 / 39.0983) +  # Z/M for K
            w_I * (53 / 126.90447) +  # Z/M for I
            w_H * (1 / 1.00784) +  # Z/M for H
            w_O * (8 / 15.999)  # Z/M for O
    )

    return e_density


# Setup
font_path = '/Users/dominikjohn/Library/Fonts/TUMNeueHelvetica-Regular.ttf'
fm.findfont(fm.FontProperties(fname=font_path))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'TUM Neue Helvetica'
plt.rcParams.update({'font.size': 12})

rho_KI = 3.13
rho_w = 1.00
E = 30

# Molecular masses
M_w = 18.01528
M_KI = 166.0028

percentages = np.linspace(0, 20, 11)
mu_total = []
e_densities = []

for p in percentages:
    w_KI = p / 100
    w_w = 1 - w_KI
    rho_mix = w_KI * rho_KI + w_w * rho_w

    # Mass fractions
    w_H = w_w * ((2 * 1.008) / M_w)
    w_O = w_w * (15.999 / M_w)
    w_K = w_KI * (39.0983 / M_KI)
    w_I = w_KI * (126.90447 / M_KI)

    # Calculate attenuation
    mu = (w_K * xraylib.CS_Total(19, E) +  # K
          w_I * xraylib.CS_Total(53, E) +  # I
          w_H * xraylib.CS_Total(1, E) +  # H
          w_O * xraylib.CS_Total(8, E)  # O
          ) * rho_mix
    mu_total.append(mu)

    # Calculate electron density
    e_density = calculate_electron_density(w_K, w_I, w_H, w_O, rho_mix)
    e_densities.append(e_density)

# Pure water calculations
w_H_pure = (2 * 1.008) / M_w
w_O_pure = 15.999 / M_w
mu_w = (w_H_pure * xraylib.CS_Total(1, E) +
        w_O_pure * xraylib.CS_Total(8, E)) * rho_w
e_density_water = calculate_electron_density(0, 0, w_H_pure, w_O_pure, rho_w)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Attenuation plot
ax1.plot(percentages, mu_total)
ax1.axhline(y=mu_w, color='g', linestyle='--', label='Pure water')
ax1.set_xlabel('KI content (wt%)')
ax1.set_ylabel('Linear attenuation coefficient µ [1/cm]')
ax1.set_title(f'Attenuation of KI-water mixture at {E} keV')
ax1.legend()
ax1.grid(True)

# Electron density plot
ax2.plot(percentages, e_densities)
ax2.axhline(y=e_density_water, color='g', linestyle='--', label='Pure water')
ax2.set_xlabel('KI content (wt%)')
ax2.set_ylabel('Electron density [e⁻/nm^3]')
ax2.set_title('Electron density of KI-water mixture')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print results
test_points = [1, 2.5, 5, 10]
mu_test_points = np.interp(test_points, percentages, mu_total)
e_density_test_points = np.interp(test_points, percentages, e_densities)

print("\nResults at test points:")
print("=" * 50)
for conc, mu, ed in zip(test_points, mu_test_points, e_density_test_points):
    print(f"\nAt {conc:>4}% KI:")
    print(f"µ = {mu:.2f} 1/cm")
    print(f"Electron density = {ed:.2f} e⁻/nm^3")
    print(f"Relative to water (µ): {(mu / mu_w - 1) * 100:.1f}%")
    print(f"Relative to water (e⁻): {(ed / e_density_water - 1) * 100:.1f}%")

# Calculate relative differences between consecutive points
rel_diffs_mu = np.diff(mu_test_points) / mu_test_points[:-1] * 100
print("\nRelative increases in attenuation:")
print("=" * 50)
for i, diff in enumerate(rel_diffs_mu):
    print(f"From {test_points[i]}% to {test_points[i + 1]}%: {diff:.1f}%")