import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle


def load_csv_to_numpy(filename):
    """Load space-delimited file and convert to numpy arrays."""
    try:
        # Try loading as space-delimited file with no header
        df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['wavelength', 'value'])
        return df['wavelength'].values, df['value'].values
    except Exception as e:
        print(f"Error with space-delimited format, trying standard CSV: {e}")
        try:
            # Fall back to standard CSV format
            df = pd.read_csv(filename)
            if len(df.columns) >= 2:
                return df.iloc[:, 0].values, df.iloc[:, 1].values
            else:
                raise ValueError(f"File {filename} doesn't have at least 2 columns")
        except Exception as e2:
            print(f"Error loading file {filename}: {e2}")
            raise


def interpolate_to_target_wavelengths(source_wavelengths, source_values, target_wavelengths):
    """Interpolate source data to match target wavelengths."""
    f = interpolate.interp1d(source_wavelengths, source_values,
                             bounds_error=False, fill_value=0.0)

    # Apply to target wavelengths
    interpolated_values = f(target_wavelengths)
    return interpolated_values


def importer(base_path, show_plots=False):
    white_wavelengths, white_intensities = load_csv_to_numpy(os.path.join(base_path, 'white-led-spectrum.csv'))
    print(f"Loaded {len(white_wavelengths)} data points for white LED spectrum")

    h_wavelengths, h_absorptions = load_csv_to_numpy(os.path.join(base_path, 'figure17curve2.csv'))
    print(f"Loaded {len(h_wavelengths)} data points for haematoxylin")

    min_wavelength = np.min(white_wavelengths)
    max_wavelength = np.max(white_wavelengths)
    print(f"White LED spectrum range: {min_wavelength} to {max_wavelength} nm")

    h_interpolated = interpolate_to_target_wavelengths(h_wavelengths, h_absorptions, white_wavelengths)

    # Convert haematoxylin absorption to molar extinction coefficient
    h_interpolated /= 1.72e-5

    if show_plots:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(white_wavelengths, white_intensities, 'b-', label='White LED')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('White LED Spectrum')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(white_wavelengths, h_interpolated, 'purple', label='Haematoxylin')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('$\epsilon$ [l/(mol$\cdot$cm)]')
        plt.title('Interpolated Stain Spectra')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        output_png_path = os.path.join(base_path, 'spectra_visualization.png')
        plt.savefig(output_png_path)

    return {
        'wavelengths': white_wavelengths,
        'white_led': white_intensities,
        'haematoxylin': h_interpolated,
    }


class ColourSystem:
    """A class representing a colour system."""

    def __init__(self, red, green, blue, white, cmf_data=None):
        """Initialise the ColourSystem object."""
        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white

        # Store CMF data
        self.cmf_data = cmf_data
        if cmf_data is not None:
            self.cmf_wavelengths = cmf_data[:, 0]
            self.cmf = cmf_data[:, 1:4]  # Just the xyz columns
        else:
            self.cmf_wavelengths = None
            self.cmf = None

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None, clip_negative=True, gamma_correct=True):
        """Transform from xyz to rgb representation of colour."""
        # Apply transformation matrix to convert XYZ to RGB
        rgb = self.T.dot(xyz)

        if clip_negative:
            # Handle out-of-gamut colors by clipping
            rgb = np.maximum(rgb, 0)

        if gamma_correct:
            # Apply gamma correction (standard sRGB gamma)
            rgb_gamma = np.where(
                rgb <= 0.0031308,
                12.92 * rgb,
                1.055 * np.power(rgb, 1 / 2.4) - 0.055
            )
            rgb = rgb_gamma

        # Clip to ensure we're in [0,1] range
        rgb = np.clip(rgb, 0, 1)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""
        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, wavelengths, spec, normalize=True):
        """Convert a spectrum to an xyz point."""
        # Check if we need to interpolate the spectrum to match CMF wavelengths
        if (self.cmf_wavelengths is not None and
                (len(wavelengths) != len(self.cmf_wavelengths) or
                 not np.array_equal(wavelengths, self.cmf_wavelengths))):

            # Interpolate spectrum to match CMF wavelengths
            interpolator = interpolate.interp1d(
                wavelengths, spec,
                bounds_error=False, fill_value=0.0)

            # Get spectrum values at CMF wavelengths
            interpolated_spec = interpolator(self.cmf_wavelengths)
        else:
            interpolated_spec = spec

        # Calculate XYZ using the color matching functions
        XYZ = np.sum(interpolated_spec[:, np.newaxis] * self.cmf, axis=0)

        # Normalize if requested (for color quality only)
        if normalize:
            den = np.sum(XYZ)
            if den == 0.:
                return XYZ
            return XYZ / den
        else:
            return XYZ

    def spec_to_rgb(self, wavelengths, spec, out_fmt=None, clip_negative=True, gamma_correct=True, normalize=True):
        """Convert a spectrum to an rgb value."""
        xyz = self.spec_to_xyz(wavelengths, spec, normalize=normalize)
        return self.xyz_to_rgb(xyz, out_fmt, clip_negative, gamma_correct)


def calculate_transmitted_spectrum(wavelengths, extinction_coefficients,
                                   thickness_um=100, concentration=50e-4,
                                   white_spectrum=None):
    """Calculate transmitted light spectrum using Beer-Lambert law."""
    # Convert thickness from µm to cm for Beer-Lambert law
    thickness_cm = thickness_um * 1e-4

    # Calculate absorbance using Beer-Lambert law: A = ε * c * l
    absorbance = extinction_coefficients * concentration * thickness_cm

    # Calculate transmittance: T = 10^(-A)
    transmittance = 10 ** (-absorbance)

    # Source light spectrum (if not provided, use white light)
    if white_spectrum is None:
        source_spectrum = np.ones_like(wavelengths)
    else:
        source_spectrum = white_spectrum

    # Calculate transmitted light
    transmitted_spectrum = source_spectrum * transmittance

    return {
        'wavelengths': wavelengths,
        'transmittance': transmittance,
        'transmitted_spectrum': transmitted_spectrum,
        'avg_transmittance': np.mean(transmittance)
    }


def get_color_with_transmittance(wavelengths, spectrum, avg_transmittance, cs):
    """Get RGB color scaled by transmittance."""
    # Get the chromaticity (normalized color) - this ignores brightness
    rgb_normalized = cs.spec_to_rgb(
        wavelengths,
        spectrum,
        clip_negative=True,
        gamma_correct=True,
        normalize=True  # Normalize to get pure color quality
    )

    # Scale RGB by the average transmittance to account for brightness
    rgb_scaled = rgb_normalized * avg_transmittance

    # Convert to hex
    return cs.rgb_to_hex(rgb_scaled)


def visualize_absorption_simple(wavelengths, extinction_coefficients, base_path, max_thickness=500, num_samples=25, concentration=10e-4):
    """Visualize haematoxylin color at different thickness levels using simple scaling."""
    # Load color matching functions
    cmf_data = np.loadtxt(os.path.join(base_path, 'cie-cmf.txt'))

    # Set up color system for sRGB
    illuminant_D65 = np.array([0.3127, 0.3291, 0.3582])  # D65 white point
    cs_srgb = ColourSystem(
        red=np.array([0.64, 0.33, 0.03]),
        green=np.array([0.30, 0.60, 0.10]),
        blue=np.array([0.15, 0.06, 0.79]),
        white=illuminant_D65,
        cmf_data=cmf_data
    )

    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 7))

    # First calculate the color at zero thickness (no absorption) for reference
    zero_thickness_result = calculate_transmitted_spectrum(
        wavelengths,
        extinction_coefficients,
        thickness_um=0,
        concentration=concentration
    )

    zero_rgb = cs_srgb.spec_to_rgb(
        wavelengths,
        zero_thickness_result['transmitted_spectrum'],
        normalize=True  # Get the pure color
    )

    # Loop through thickness samples
    for i in range(num_samples):
        # Calculate thickness for this sample
        thickness_um = i * (max_thickness / (num_samples - 1))

        # Calculate transmitted spectrum
        result = calculate_transmitted_spectrum(
            wavelengths,
            extinction_coefficients,
            thickness_um=thickness_um,
            concentration=concentration
        )

        # Get average transmittance
        avg_transmittance = result['avg_transmittance']

        # Get the HTML color code
        # Method 1: Calculate normalized color, then scale by transmittance
        if avg_transmittance > 0.01:  # Avoid division by zero for nearly black colors
            html_rgb = get_color_with_transmittance(
                wavelengths,
                result['transmitted_spectrum'],
                avg_transmittance,
                cs_srgb
            )
        else:
            # Nearly black for very low transmittance
            html_rgb = '#000000'

        # Place and label a circle with the color
        x, y = i % 6, -(i // 6)
        circle = Circle(xy=(x, y * 1.2), radius=0.4, fc=html_rgb)
        ax.add_patch(circle)

        # Add label with thickness and transmittance
        ax.annotate(f'{thickness_um:.0f} µm\nT={avg_transmittance:.2f}',
                    xy=(x, y * 1.2),
                    va='center', ha='center',
                    color='white' if avg_transmittance < 0.5 else 'black',
                    fontsize=8)

    # Set the limits and background color; remove the ticks
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-4.35, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('k')
    ax.set_title('Haematoxylin Color vs. Thickness (Simple Scaling)', color='white', size=14)

    # Make sure our circles are circular
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_transmittance_curve(wavelengths, extinction_coefficients, concentration=500e-4):
    """Plot the transmittance curve for different thicknesses."""
    thicknesses = np.linspace(0, 500, 100)
    avg_transmittances = []

    for thickness in thicknesses:
        result = calculate_transmitted_spectrum(
            wavelengths,
            extinction_coefficients,
            thickness_um=thickness,
            concentration=concentration
        )
        avg_transmittance = result['avg_transmittance']
        avg_transmittances.append(avg_transmittance)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thicknesses, avg_transmittances, 'b-', linewidth=2)
    ax.set_xlabel('Thickness (µm)')
    ax.set_ylabel('Average Transmittance')
    ax.set_title('Haematoxylin Light Transmission vs. Thickness')
    ax.grid(True)
    ax.set_ylim(0, 1)

    # Add reference line at 50% transmittance
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    ax.text(thicknesses[-1] * 0.8, 0.52, 'T = 0.5', color='r')

    # Add 10% transmittance line
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7)
    ax.text(thicknesses[-1] * 0.8, 0.12, 'T = 0.1', color='r')

    return fig


def main():
    base_path = '/Users/dominikjohn/Library/Mobile Documents/com~apple~CloudDocs/Documents/1_Projects/Paper Material Decomposition/visiblelight'

    # Import haematoxylin data
    result_dict = importer(base_path)
    wavelengths = result_dict['wavelengths']
    hematin = result_dict['haematoxylin']

    # Create the new wavelength grid (380 to 780 in steps of 5)
    new_wavelengths = np.arange(380, 785, 5)

    # Create an interpolation function
    interpolator = interpolate.interp1d(wavelengths, hematin, bounds_error=False, fill_value=0)

    # Get interpolated values at the new wavelength points
    new_hematin = interpolator(new_wavelengths)

    # Display the interpolated haematoxylin data
    plt.figure()
    plt.plot(new_wavelengths, new_hematin, label='Haematoxylin')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Extinction Coefficient')
    plt.title('Haematoxylin Absorption Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()

    concentration = 10e-4

    fig1 = visualize_absorption_simple(new_wavelengths, new_hematin, base_path, concentration=concentration)
    plt.show()

    # Plot transmittance vs. thickness curve
    fig2 = plot_transmittance_curve(new_wavelengths, new_hematin, concentration=concentration)
    plt.show()

    print("Visualization complete.")

if __name__ == "__main__":
    main()