#!/usr/bin/env python3
"""
Calculates the X-ray attenuation coefficient of a hematein-lead complex using xraylib.
"""

import numpy as np
import matplotlib.pyplot as plt
import xraylib

# Define the atomic composition of hematein (C16H12O6)
# and the hematein-lead complex (C16H12O6-Pb)

# Atomic numbers for reference
C_Z = 6  # Carbon
H_Z = 1  # Hydrogen
O_Z = 8  # Oxygen
Pb_Z = 82  # Lead

# Hematein-lead complex composition (C16H12O6-Pb)
complex_composition = {
    C_Z: 16,  # 16 Carbon atoms
    H_Z: 12,  # 12 Hydrogen atoms
    O_Z: 6,  # 6 Oxygen atoms
    Pb_Z: 1  # 1 Lead atom
}

# Molecular weight calculation
complex_mw = 16 * xraylib.AtomicWeight(C_Z) + \
             12 * xraylib.AtomicWeight(H_Z) + \
             6 * xraylib.AtomicWeight(O_Z) + \
             1 * xraylib.AtomicWeight(Pb_Z)

print(f"Calculated molecular weight: {complex_mw:.2f} g/mol")

# Density of the complex in g/cm³ (from our previous calculation)
complex_density = 2.1  # g/cm³

# Calculate mass fractions for each element
mass_fractions = {}
for element, count in complex_composition.items():
    element_mass = count * xraylib.AtomicWeight(element)
    mass_fraction = element_mass / complex_mw
    mass_fractions[element] = mass_fraction
    element_name = xraylib.AtomicNumberToSymbol(element)
    print(f"{element_name}: {mass_fraction * 100:.2f}%")

# Define energy range (in keV) to calculate attenuation coefficients
energies = np.linspace(1, 100, 500)  # 1 to 100 keV

# Calculate mass attenuation coefficients (cm²/g) for the complex
# and convert to linear attenuation coefficients (cm⁻¹)
mass_att_coeffs = np.zeros_like(energies)
for energy in range(len(energies)):
    for element, fraction in mass_fractions.items():
        mass_att_coeffs[energy] += fraction * xraylib.CS_Total(element, energies[energy])

# Convert mass attenuation coefficients to linear attenuation coefficients
linear_att_coeffs = mass_att_coeffs * complex_density

# Calculate transmission for a 1mm thick sample
thickness_mm = 1.0
thickness_cm = thickness_mm / 10.0
transmission = np.exp(-linear_att_coeffs * thickness_cm)

# Create plots
plt.figure(figsize=(12, 10))

# Plot 1: Mass attenuation coefficient vs energy
plt.subplot(2, 1, 1)
plt.loglog(energies, mass_att_coeffs)
plt.xlabel('Photon Energy (keV)')
plt.ylabel('Mass Attenuation Coefficient (cm²/g)')
plt.title('Mass Attenuation Coefficient vs Photon Energy for Hematein-Lead Complex')
plt.grid(True, which="both", ls="--")

# Plot 2: Linear attenuation coefficient vs energy
plt.subplot(2, 1, 2)
plt.loglog(energies, linear_att_coeffs)
plt.xlabel('Photon Energy (keV)')
plt.ylabel('Linear Attenuation Coefficient (cm⁻¹)')
plt.title(f'Linear Attenuation Coefficient vs Photon Energy (ρ = {complex_density} g/cm³)')
plt.grid(True, which="both", ls="--")

plt.tight_layout()

# Save the plot
plt.savefig('hematein_lead_attenuation.png', dpi=300)

# Output specific values at key energies
key_energies = [25]
print("\nAttenuation Coefficients at Key Energies:")
print("Energy (keV) | Mass Att. (cm²/g) | Linear Att. (cm⁻¹) | Transmission (1mm)")
print("-" * 75)

for e in key_energies:
    # Find closest energy in our array
    idx = np.abs(energies - e).argmin()
    energy = energies[idx]
    mass_att = mass_att_coeffs[idx]
    linear_att = linear_att_coeffs[idx]
    trans = transmission[idx]
    print(f"{energy:11.1f} | {mass_att:17.4f} | {linear_att:17.4f} | {trans:19.4f}")

# Compare with pure lead for reference
print("\nComparison with Pure Lead at Key Energies:")
print("Energy (keV) | Complex Linear Att. (cm⁻¹) | Pure Pb Linear Att. (cm⁻¹) | Ratio")
print("-" * 80)

lead_density = 11.34  # g/cm³
for e in key_energies:
    idx = np.abs(energies - e).argmin()
    energy = energies[idx]
    complex_att = linear_att_coeffs[idx]

    # Calculate for pure lead
    lead_mass_att = xraylib.CS_Total(Pb_Z, energy)
    lead_linear_att = lead_mass_att * lead_density

    ratio = complex_att / lead_linear_att
    print(f"{energy:11.1f} | {complex_att:27.4f} | {lead_linear_att:27.4f} | {ratio:5.4f}")

print("\nNote: This calculation assumes a homogeneous material with the composition of a 1:1")
print("hematein-lead complex and a density of 2.1 g/cm³. The actual attenuation may vary")
print("depending on the exact molecular structure and packing of the complex.")