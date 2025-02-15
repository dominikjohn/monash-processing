import xraylib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Register font and set it as default
font_path = '/Users/dominikjohn/Library/Fonts/TUMNeueHelvetica-Regular.ttf'
fm.findfont(fm.FontProperties(fname=font_path))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'TUM Neue Helvetica'
plt.rcParams.update({'font.size': 12})

# Rest of your code exactly as it was originally
rho_KI = 3.13
rho_eth = 0.789
E = 25

# Molecular masses
M_eth = 46.069  # C2H5OH
M_KI = 166.0028

percentages = np.linspace(0, 100, 11)
mu_total = []

for p in percentages:
    w_KI = p / 100
    w_eth = 1 - w_KI
    rho_mix = w_KI * rho_KI + w_eth * rho_eth

    # Corrected mass fractions within ethanol portion
    w_C = w_eth * ((2 * 12.011) / M_eth)
    w_H = w_eth * ((6 * 1.008) / M_eth)
    w_O = w_eth * (15.999 / M_eth)

    # KI mass fractions
    w_K = w_KI * (39.0983 / M_KI)
    w_I = w_KI * (126.90447 / M_KI)

    mu = (w_K * xraylib.CS_Total(19, E) +  # K
          w_I * xraylib.CS_Total(53, E) +  # I
          w_C * xraylib.CS_Total(6, E) +  # C
          w_H * xraylib.CS_Total(1, E) +  # H
          w_O * xraylib.CS_Total(8, E)  # O
          ) * rho_mix

    mu_total.append(mu)

# Calculate pure ethanol mu with correct mass fractions
w_C_pure = (2 * 12.011) / M_eth
w_H_pure = (6 * 1.008) / M_eth
w_O_pure = 15.999 / M_eth

mu_eth = (w_C_pure * xraylib.CS_Total(6, E) +
          w_H_pure * xraylib.CS_Total(1, E) +
          w_O_pure * xraylib.CS_Total(8, E)) * rho_eth

plt.plot(percentages, mu_total)
plt.axhline(y=mu_eth, color='g', linestyle='--', label='Pure ethanol')
plt.axvline(x=2.23, color='r', linestyle='--', label='Max. KI solubility at 25°C')
plt.xlabel('KI content (wt%)')
plt.ylabel('Linear attenuation coefficient µ [1/cm]')
plt.title('Attenuation of KI-ethanol mixture at 25 keV')
plt.legend()
plt.grid(True)
plt.show()

m_KI = 1.8  # g
m_ethanol = rho_eth * 100 # g (100 mL * 1.00 g/mL)
mass_percent = (m_KI / (m_KI + m_ethanol)) * 100
print(f"\nMaximum solubility: {mass_percent:.2f}%")