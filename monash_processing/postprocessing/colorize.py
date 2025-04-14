import numpy as np
import pandas as pd
from scipy import interpolate
import os
from scipy.constants import h, c, k
import matplotlib.pyplot as plt

class Colorizer:

    def __init__(self, base_path):
        self.base_path = base_path

    def concentration_to_color(self, wavelengths, epsilon, concentration, thickness_um, light_color=6500):
        # Load color matching functions
        cmf_data = np.loadtxt(os.path.join(self.base_path, 'cie-cmf.txt'))

        illuminant_D65 = np.array([0.3127, 0.3291, 0.3582])
        cs_srgb = ColourSystem(
            red=np.array([0.64, 0.33, 0.03]),
            green=np.array([0.30, 0.60, 0.10]),
            blue=np.array([0.15, 0.06, 0.79]),
            white=illuminant_D65,
            cmf_data=cmf_data
        )

        result = self.calculate_transmitted_spectrum(
            wavelengths,
            epsilon,
            thickness_um=thickness_um,
            concentration=concentration,
            light_color=light_color
        )

        trans = result['transmitted_spectrum']
        shape = result['shape']

        # Build color array
        hex_colors = []
        for i in range(len(trans)):
            rgb = cs_srgb.spec_to_rgb(
                wavelengths,
                trans[i],
                clip_negative=True,
                gamma_correct=True,
            )
            hex_color = cs_srgb.rgb_to_hex(rgb)
            hex_colors.append(hex_color)

        return np.array(hex_colors).reshape(shape)

    def calculate_transmitted_spectrum(self, wavelengths, epsilon,
                                       thickness_um, concentration, light_color):
        """Vectorized version of transmitted spectrum using Beer-Lambert law."""

        # Ensure arrays
        concentration = np.atleast_1d(concentration)
        concentration = np.maximum(concentration, 0.0)
        thickness_um = np.atleast_1d(thickness_um)

        # Concentration is in M, which is mol/L

        concentration, thickness_um = np.broadcast_arrays(concentration, thickness_um)
        shape = concentration.shape

        # Convert to cm
        thickness_cm = thickness_um * 1e-4

        # ε shape: (λ,)
        # concentration/thickness shape: (N,)
        # absorbance shape: (N, λ)
        absorbance = np.outer(concentration * thickness_cm, epsilon)

        # Transmittance: T = 10^(-A)
        transmittance = 10 ** (-absorbance)

        # Source spectrum: Planck at light_color K
        source_spectrum = self.planck(wavelengths, light_color)  # shape: (λ,)

        # Transmitted spectrum: (N, λ)
        transmitted_spectrum = transmittance * source_spectrum

        return {
            'wavelengths': wavelengths,
            'transmittance': transmittance,
            'transmitted_spectrum': transmitted_spectrum,
            'source_spectrum': source_spectrum,
            'shape': shape
        }

    @staticmethod
    def planck(lam, T):
        """ Returns the spectral radiance of a black body at temperature T.

        Returns the spectral radiance, B(lam, T), in W.sr-1.m-2 of a black body
        at temperature T (in K) at a wavelength lam (in nm), using Planck's law.

        """
        lam_m = lam / 1.e9
        fac = h * c / lam_m / k / T
        B = 2 * h * c ** 2 / lam_m ** 5 / (np.exp(fac) - 1)
        return B

    def interpolate_to_target_wavelengths(self, source_wavelengths, source_values, min_wavelength, max_wavelength,
                                          step):
        """Interpolate source data to match target wavelengths with specified step size."""
        target_wavelengths = np.arange(min_wavelength, max_wavelength + step, step)
        f = interpolate.interp1d(source_wavelengths, source_values,
                                 bounds_error=False, fill_value=0.0)
        interpolated_values = f(target_wavelengths)
        return interpolated_values, target_wavelengths

    def import_absorbances(self):
        h_wavelengths, h_absorptions = self.load_csv_to_numpy(os.path.join(self.base_path, 'figure17curve2.csv'))
        min_wavelength = 380
        max_wavelength = 780
        h_interpolated, target_wavelengths = self.interpolate_to_target_wavelengths(h_wavelengths, h_absorptions,
                                                                                    min_wavelength, max_wavelength, 5)

        return {
            'absorbances': h_interpolated,
            'wavelengths': target_wavelengths
        }

    def load_csv_to_numpy(self, filename):
        try:
            df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['wavelength', 'value'])
            return df['wavelength'].values, df['value'].values
        except Exception as e:
            print(f"Error with space-delimited format, trying standard CSV: {e}")
            df = pd.read_csv(filename)
            if len(df.columns) >= 2:
                return df.iloc[:, 0].values, df.iloc[:, 1].values
            else:
                raise ValueError("CSV file does not contain at least two columns.")

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

    def compute_luminance(self, spectrum):
        """Compute luminance (Y value) using the CMF y-bar curve."""
        y_bar = self.cmf[:, 1]
        return np.sum(spectrum * y_bar)

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

    def spec_to_xyz(self, wavelengths, spec, normalize=False):
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

    def spec_to_rgb(self, wavelengths, spec, out_fmt=None, clip_negative=True, gamma_correct=True, normalize=False):
        """Convert a spectrum to an rgb value."""
        xyz = self.spec_to_xyz(wavelengths, spec, normalize=normalize)
        return self.xyz_to_rgb(xyz, out_fmt, clip_negative, gamma_correct)