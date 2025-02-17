import xraylib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/Users/dominikjohn/Library/Fonts/TUMNeueHelvetica-Regular.ttf'
fm.findfont(fm.FontProperties(fname=font_path))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'TUM Neue Helvetica'
plt.rcParams.update({'font.size': 12})

rho_KI = 3.13
rho_w = 1.00
E = 35

# Molecular masses
M_w = 18.01528  # C2H5OH
M_KI = 166.0028

percentages = np.linspace(0, 20, 11)
mu_total = []

for p in percentages:
    w_KI = p / 100
    w_w = 1 - w_KI
    rho_mix = w_KI * rho_KI + w_w * rho_w

    # Corrected mass fractions within ethanol portion
    w_H = w_w * ((2 * 1.008) / M_w)
    w_O = w_w * (15.999 / M_w)

    # KI mass fractions
    w_K = w_KI * (39.0983 / M_KI)
    w_I = w_KI * (126.90447 / M_KI)

    mu = (w_K * xraylib.CS_Total(19, E) +  # K
          w_I * xraylib.CS_Total(53, E) +  # I
          w_H * xraylib.CS_Total(1, E) +  # H
          w_O * xraylib.CS_Total(8, E)  # O
          ) * rho_mix

    mu_total.append(mu)

# Calculate pure ethanol mu with correct mass fractions
w_H_pure = (2 * 1.008) / M_w
w_O_pure = 15.999 / M_w

mu_w = (w_H_pure * xraylib.CS_Total(1, E) +
          w_O_pure * xraylib.CS_Total(8, E)) * rho_w

plt.plot(percentages, mu_total)
plt.axhline(y=mu_w, color='g', linestyle='--', label='Pure water')
#plt.axvline(x=58.33, color='r', linestyle='--', label='Max. KI solubility at 25°C')
plt.xlabel('KI content (wt%)')
plt.ylabel('Linear attenuation coefficient µ [1/cm]')
plt.title(f'Attenuation of KI-water mixture at {E} keV')
plt.legend()
plt.grid(True)
plt.show()

# Print the maximum solubility point
m_KI = 140  # g
m_water = 100  # g (100 mL * 1.00 g/mL)
mass_percent = (m_KI / (m_KI + m_water)) * 100
print(f"\nMaximum solubility: {mass_percent:.2f}%")

test_points = [1, 2.5, 5, 10]  # concentrations we want to check

# Use numpy's interp to get values at our test points from the existing simulation
mu_test_points = np.interp(test_points, percentages, mu_total)

# Print results
for conc, mu in zip(test_points, mu_test_points):
    print(f"At {conc:>4}% KI: µ = {mu:.2f} cm⁻¹")

# Calculate the relative differences between consecutive points
rel_diffs = np.diff(mu_test_points) / mu_test_points[:-1] * 100
rel_diffs_water = mu_test_points / mu_w * 100
for i, diff in enumerate(rel_diffs):
    print(f"Relative increase from {test_points[i]}% to {test_points[i+1]}%: {diff:.1f}%")
    print(f"Relative to water ", rel_diffs_water[i], "%")