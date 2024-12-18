def calculate_ethanol_mixture_properties(concentration_by_volume):
    """
    Calculate properties of ethanol-water mixture at given concentration.

    Args:
        concentration_by_volume (float): Volume fraction of ethanol (0-1)

    Returns:
        dict: Material properties for the mixture
    """
    # Properties of pure components
    ethanol = {
        'density': 0.789,  # g/cm³
        'molecular_weight': 46.068,  # g/mol
        'electrons': 26,
        'composition': {'C': 2, 'H': 6, 'O': 1}
    }

    water = {
        'density': 1.0,  # g/cm³
        'molecular_weight': 18.015,  # g/mol
        'electrons': 10,
        'composition': {'H': 2, 'O': 1}
    }

    # Calculate mass fraction from volume fraction
    volume_ethanol = concentration_by_volume
    volume_water = 1 - concentration_by_volume

    mass_ethanol = volume_ethanol * ethanol['density']
    mass_water = volume_water * water['density']
    total_mass = mass_ethanol + mass_water

    mass_fraction_ethanol = mass_ethanol / total_mass
    mass_fraction_water = mass_water / total_mass

    # Calculate effective properties
    effective_density = total_mass / (volume_ethanol + volume_water)

    # Calculate effective molecular weight (weighted average)
    effective_mw = (mass_fraction_ethanol * ethanol['molecular_weight'] +
                    mass_fraction_water * water['molecular_weight'])

    # Calculate effective number of electrons (weighted average)
    effective_electrons = (mass_fraction_ethanol * ethanol['electrons'] +
                           mass_fraction_water * water['electrons'])

    # Combine compositions with proper scaling
    effective_composition = {}

    # Add ethanol contribution
    for element, count in ethanol['composition'].items():
        effective_composition[element] = count * mass_fraction_ethanol

    # Add water contribution
    for element, count in water['composition'].items():
        if element in effective_composition:
            effective_composition[element] += count * mass_fraction_water
        else:
            effective_composition[element] = count * mass_fraction_water

    return {
        'density': effective_density,
        'molecular_weight': effective_mw,
        'electrons': effective_electrons,
        'composition': effective_composition
    }


# Example usage:
concentration = 0.96  # 40% ethanol by volume
ethanol_mixture = calculate_ethanol_mixture_properties(concentration)

# Print the results
print(f"Properties of {concentration * 100}% ethanol mixture:")
print(f"Density: {ethanol_mixture['density']:.3f} g/cm³")
print(f"Molecular Weight: {ethanol_mixture['molecular_weight']:.3f} g/mol")
print(f"Electrons: {ethanol_mixture['electrons']:.1f}")
print("Composition:", ethanol_mixture['composition'])