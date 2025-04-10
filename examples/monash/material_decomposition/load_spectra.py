import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import os

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

    #e_wavelengths, e_absorptions = load_csv_to_numpy(os.path.join(base_path, 'eosin.csv'))
    #print(f"Loaded {len(e_wavelengths)} data points for eosin")

    min_wavelength = np.min(white_wavelengths)
    max_wavelength = np.max(white_wavelengths)
    print(f"White LED spectrum range: {min_wavelength} to {max_wavelength} nm")

    h_interpolated = interpolate_to_target_wavelengths(h_wavelengths, h_absorptions, white_wavelengths)
    #e_interpolated = interpolate_to_target_wavelengths(e_wavelengths, e_absorptions, white_wavelengths)

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
        #plt.plot(white_wavelengths, h_interpolated, 'purple', label='Haematoxylin')
        plt.plot(white_wavelengths, e_interpolated, 'pink', label='Eosin')
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
        #'eosin': e_interpolated
    }

base_path = '/Users/dominikjohn/Library/Mobile Documents/com~apple~CloudDocs/Documents/1_Projects/Paper Material Decomposition/visiblelight'
result_dict = importer(base_path)
from scipy.interpolate import interp1d

wavelengths = result_dict['wavelengths']
hematin = result_dict['haematoxylin']

# Create the new wavelength grid (380 to 780 in steps of 5)
new_wavelengths = np.arange(380, 785, 5)

# Create an interpolation function
interpolator = interp1d(wavelengths, hematin, bounds_error=False, fill_value=0)

# Get interpolated values at the new wavelength points
new_hematin = interpolator(new_wavelengths)
plt.figure()
plt.plot(new_wavelengths, new_hematin, label='Haematoxylin')
plt.show()


def calculate_transmitted_spectrum(wavelengths, extinction_coefficients, thickness_um=100, concentration=50e-4):
    # Convert thickness from µm to cm for Beer-Lambert law
    thickness_cm = thickness_um * 1e-4

    # Calculate absorbance using Beer-Lambert law: A = ε * c * l
    absorbance = extinction_coefficients * concentration * thickness_cm

    # Calculate transmittance: T = 10^(-A)
    transmittance = 10 ** (-absorbance)

    # Source light spectrum (white light - equal intensity at all wavelengths)
    source_spectrum = np.ones_like(wavelengths)

    # Calculate transmitted light
    transmitted_spectrum = source_spectrum * transmittance

    return {
        'wavelengths': wavelengths,
        'transmittance': transmittance,
        'transmitted_spectrum': transmitted_spectrum,
    }


def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    This version includes improved handling of color brightness.
    """

    def __init__(self, red, green, blue, white, cmf):
        """Initialise the ColourSystem object."""
        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        self.cmf = cmf  # Pass the color matching function
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None, clip_negative=True, gamma_correct=True):
        """Transform from xyz to rgb representation of colour.

        This improved version:
        1. Uses better handling for out-of-gamut colors
        2. Allows preserving absolute brightness
        3. Includes gamma correction
        """
        # Apply transformation matrix to convert XYZ to RGB
        rgb = self.T.dot(xyz)

        if clip_negative:
            # Handle out-of-gamut colors by clipping - this is simpler and
            # better preserves darkness than desaturation for our application
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

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point."""
        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None, clip_negative=True, gamma_correct=True):
        """Convert a spectrum to an rgb value."""
        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt, clip_negative, gamma_correct)


# ----- Beer-Lambert Law for light absorption -------

def calculate_transmitted_spectrum(wavelengths, extinction_coefficients,
                                   thickness_um=100, concentration=50e-4,
                                   white_spectrum=None):
    """Calculate transmitted light spectrum using Beer-Lambert law.

    Parameters:
    - wavelengths: array of wavelength values
    - extinction_coefficients: molar extinction coefficient at each wavelength
    - thickness_um: thickness in micrometers
    - concentration: molar concentration
    - white_spectrum: optional source spectrum (defaults to flat white)

    Returns:
    - Dictionary with wavelengths, transmittance and transmitted spectrum
    """
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
    }

max_thickness = 500
num_samples = 25

# Set up color system
illuminant_D65 = np.array([0.3127, 0.3291, 0.3582])  # D65 white point
cs_srgb = ColourSystem(
    red=np.array([0.64, 0.33, 0.03]),
    green=np.array([0.30, 0.60, 0.10]),
    blue=np.array([0.15, 0.06, 0.79]),
    white=illuminant_D65,
    cmf=np.loadtxt(base_path + '/cie-cmf.txt', usecols=(1, 2, 3))

)

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Loop through thickness samples
for i in range(num_samples):
    # Calculate thickness for this sample
    thickness_um = i * (max_thickness / (num_samples - 1))

    # Calculate transmitted spectrum
    result = calculate_transmitted_spectrum(
        wavelengths, new_hematin,
        thickness_um=thickness_um,
        concentration=10e-4
    )
    transmitted_spectrum = result['transmitted_spectrum']

    # Convert spectrum to RGB color - using our improved function
    html_rgb = cs_srgb.spec_to_rgb(
        transmitted_spectrum,
        out_fmt='html',
        clip_negative=True,
        gamma_correct=True
    )

    # Place and label a circle with the colour
    x, y = i % 6, -(i // 6)
    circle = Circle(xy=(x, y * 1.2), radius=0.4, fc=html_rgb)
    ax.add_patch(circle)

    # Add label with thickness
    text_color = 'white' if i > num_samples / 3 else 'black'
    ax.annotate('{:.0f} µm'.format(thickness_um),
                xy=(x, y * 1.2 - 0.5),
                va='center', ha='center',
                color=text_color)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-4.35, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
ax.set_title('Haematoxylin Color vs. Thickness', color='white', size=14)

# Make sure our circles are circular!
ax.set_aspect("equal")
plt.tight_layout()

'''illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)
cs = cs_hdtv  # Use your preferred color system
visualize_haematoxylin_thickness(new_wavelengths, new_hematin, cs_hdtv.cmf)'''
