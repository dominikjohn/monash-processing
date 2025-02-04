import numpy as np
from scipy import special, ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tifffile

def visualize_line_on_image(image, start_point, end_point, title="Image with Line Profile"):
    """
    Visualize the line profile path on the image.

    Parameters:
    image (np.ndarray): 2D or 3D input image (if 3D, uses mean across first dimension)
    start_point (tuple): (x, y) coordinates of start point
    end_point (tuple): (x, y) coordinates of end point
    title (str): Title for the plot
    """
    # If image is 3D, take mean across first dimension
    if len(image.shape) == 3:
        display_image = np.mean(image, axis=0)
    else:
        display_image = image

    plt.figure(figsize=(10, 10))
    plt.imshow(display_image, cmap='gray')
    plt.plot([start_point[0], end_point[0]],
             [start_point[1], end_point[1]],
             'r-', linewidth=2, label='Profile Line')
    plt.plot(start_point[0], start_point[1], 'go', label='Start Point')
    plt.plot(end_point[0], end_point[1], 'ro', label='End Point')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.colorbar(label='Intensity')
    plt.axis('image')
    plt.show()


import numpy as np
from scipy import special, ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def visualize_line_on_image(image, start_point, end_point, title="Image with Line Profile"):
    """
    Visualize the line profile path on the image.

    Parameters:
    image (np.ndarray): 2D or 3D input image (if 3D, uses mean across first dimension)
    start_point (tuple): (x, y) coordinates of start point
    end_point (tuple): (x, y) coordinates of end point
    title (str): Title for the plot
    """
    # If image is 3D, take mean across first dimension
    if len(image.shape) == 3:
        display_image = np.mean(image, axis=0)
    else:
        display_image = image

    plt.figure(figsize=(10, 10))
    plt.imshow(display_image, cmap='gray')
    plt.plot([start_point[0], end_point[0]],
             [start_point[1], end_point[1]],
             'r-', linewidth=2, label='Profile Line')
    plt.plot(start_point[0], start_point[1], 'go', label='Start Point')
    plt.plot(end_point[0], end_point[1], 'ro', label='End Point')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.colorbar(label='Intensity')
    plt.axis('image')
    plt.show()


def get_line_profile(image, start_point, end_point, num_points=100):
    """
    Extract a line profile from a 2D or 3D image between two points using interpolation.
    For 3D images, averages profiles across all slices.

    Parameters:
    image (np.ndarray): 2D or 3D input image. If 3D, shape should be [slices, Y, X]
    start_point (tuple): (x, y) coordinates of start point
    end_point (tuple): (x, y) coordinates of end point
    num_points (int): Number of points to sample along the line

    Returns:
    tuple: (positions, intensities) where positions are in pixel units
    """
    # Create coordinates for line points
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)

    if len(image.shape) == 3:
        # Handle 3D image (multiple slices)
        profiles = []
        for slice_idx in range(image.shape[0]):
            profile = ndimage.map_coordinates(image[slice_idx],
                                              np.vstack((y, x)))
            profiles.append(profile)

        # Average across all slices
        line_profile = np.mean(profiles, axis=0)

        # Also calculate standard deviation for error estimation
        line_profile_std = np.std(profiles, axis=0)
    else:
        # Handle 2D image (single slice)
        line_profile = ndimage.map_coordinates(image, np.vstack((y, x)))
        line_profile_std = np.zeros_like(line_profile)

    # Calculate positions in pixel units
    positions = np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2)

    return positions, line_profile, line_profile_std


def erffunc(x, a, b, c, x_0, l):
    """
    Error function for fitting interface profiles.

    Parameters:
    x: position across line-profile
    a: vertical shift
    b: amplitude
    c: height of normalized Gaussian bumps
    x_0: translational shift
    l: width of error function
    """
    return (a + b * special.erf((x - x_0) / l) +
            c * (x - x_0) / l * np.exp(-((x - x_0) ** 2) / (l ** 2)))


def analyze_interface(image, start_point, end_point, gamma_guess, sod, odd, wavelength, det_voxel,
                      num_points=100, plot=True, visualize_line=True):
    """
    Analyze an interface in a 2D or 3D image between two points.

    Parameters:
    image (np.ndarray): 2D or 3D input image. If 3D, shape should be [slices, Y, X]
    start_point (tuple): (x, y) coordinates of start point
    end_point (tuple): (x, y) coordinates of end point
    gamma_guess (float): Initial gamma used in phase-retrieval
    sod (float): Source-to-sample distance [microns]
    odd (float): Object-to-detector distance [microns]
    wavelength (float): Wavelength of X-rays [microns]
    det_voxel (float): Detector pixel size [microns]
    num_points (int): Number of points to sample along the line
    plot (bool): Whether to show the profile plot
    visualize_line (bool): Whether to show the line overlay on image
    """
    # Validate input dimensions
    if len(image.shape) == 3:
        image = np.mean(image, axis=0)

    # Calculate magnification
    mag = (sod + odd) / sod
    pix = det_voxel / mag  # effective pixel size

    # Visualize line on image if requested
    if visualize_line:
        visualize_line_on_image(image, start_point, end_point)

    # Get line profile
    position_pixels, mu_recon, profile_std = get_line_profile(
        image, start_point, end_point, num_points)
    position_microns = position_pixels * pix

    # Convert to beta
    mu_recon_um = mu_recon * 10 ** -6
    beta_recon = (mu_recon_um * wavelength) / (4 * np.pi)

    # Initial parameter guesses
    quarter_len = len(beta_recon) // 4
    a_guess = 0.5 * (np.mean(beta_recon[:quarter_len]) +
                     np.mean(beta_recon[-quarter_len:]))
    b_guess = np.mean(beta_recon[:quarter_len]) - np.mean(beta_recon[-quarter_len:])
    c_guess = a_guess / 10
    l_guess = np.abs((np.argmin(beta_recon) - np.argmax(beta_recon))) * pix
    x_0_guess = 0.5 * np.max(position_microns)

    p0 = np.array([a_guess, b_guess, c_guess, x_0_guess, l_guess])

    # Curve fitting
    fit_coefficients, covariance = curve_fit(erffunc, position_microns, beta_recon,
                                             p0=p0, method='lm',
                                             sigma=profile_std if len(image.shape) == 3 else None)
    error = np.sqrt(np.diag(covariance))

    # Calculate fitted curve
    curvefit_data = erffunc(position_microns, *fit_coefficients)

    # Calculate true gamma
    C, B, l = fit_coefficients[2], fit_coefficients[1], fit_coefficients[4]
    tau_guess = (odd * wavelength * gamma_guess) / (mag * 4 * np.pi)
    tau_true = tau_guess + ((np.sqrt(np.pi)) * l ** 2 * C) / (4 * B)
    gamma_true = (tau_true * mag * 4 * np.pi) / (odd * wavelength)

    # Calculate relative magnitudes
    delta_relative = 2 * B * gamma_true
    beta_relative = 2 * B

    if plot:
        plt.figure(figsize=(15, 10))
        plt.title("Line Profile Analysis", fontsize=24)
        plt.errorbar(position_microns, beta_recon, yerr=profile_std if len(image.shape) == 3 else None,
                     fmt='o', label='Raw Data', alpha=0.5)
        plt.plot(position_microns, curvefit_data, 'r-',
                 linewidth=3, label='Fitted Curve')
        plt.xlabel("Position (x [\u03BCm])", fontsize=18)
        plt.ylabel("Reconstructed Attenuation Coefficient (\u03B2(x))",
                   fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=14)
        plt.show()

    results = {
        'gamma_true': gamma_true,
        'beta_relative': beta_relative,
        'delta_relative': delta_relative,
        'fit_coefficients': fit_coefficients,
        'errors': error,
        'positions': position_microns,
        'profile': beta_recon,
        'profile_std': profile_std,
        'fitted_curve': curvefit_data
    }

    return results

volume = []
for i in range(201, 202):
    volume.append(np.asarray(tifffile.imread('/data/mct/22203/results/P6_Manual/recon_att/recon_cs01175_idx_0201.tiff')))
volume = np.asarray(volume)

start_point = (1200, 990)
end_point = (1213, 1010)

start_point = (1285,1604)
end_point = (1291, 1559)

start_point = (1446,1602)
end_point = (1456,1610)

start_point = (1346, 903)
end_point = (1354,917)

start_point = (1423,829)
end_point = (1431,787)

results = analyze_interface(
    image=test3,  # or image_2d
    start_point=start_point,
    end_point=end_point,
    #gamma_guess=147.79,
    gamma_guess=1079.29,
    sod=15e6,  # 15mm in microns
    odd=0.15e6,  # 0.15mm in microns
    wavelength=5e-5,  # in microns
    det_voxel=1.444,  # in microns
    visualize_line=True  # Set to True to see line overlay
)

print(f"True gamma: {results['gamma_true']:.2f}")
print(f"Relative beta: {results['beta_relative']:.2e}")
print(f"Relative delta: {results['delta_relative']:.2e}")