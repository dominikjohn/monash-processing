from scipy.signal import savgol_filter
from scipy.interpolate import griddata


def antisym_mirror_im(im, diffaxis, mode='reflect'):
    '''
    Expands an image by mirroring it and inverting it. Can reduce artifacts in phase integration

    according to Bon 2012

    Parameters
    ----------
    im : 2D-array
        Image to be expanded

    diffaxis : str
        dx or dy
        inicates which differential is taken

    mode : str, Default='reflect'
        Mode for using np.pad(). Can be 'reflect' or 'edge'...

    Returns
    --------
    m_im : 2D-array
        Mirrored image, shape=2*im.shape()
    '''
    m_im = np.pad(im, ((im.shape[0], 0), (im.shape[1], 0)), mode=mode)

    if diffaxis == 'dx':
        m_im[:, :im.shape[1]] *= (-1)
    elif diffaxis == 'dy':
        m_im[:im.shape[0]] *= (-1)
    else:
        raise ValueError('unknown differential, please select dx or dy')

    return m_im


def pshift(a, ctr):
    """\
    Shift an array so that ctr becomes the origin.
    """
    sh = np.array(a.shape)
    out = np.zeros_like(a)

    ctri = np.floor(ctr).astype(int)
    ctrx = np.empty((2, a.ndim))
    ctrx[1, :] = ctr - ctri  # second weight factor
    ctrx[0, :] = 1 - ctrx[1, :]  # first  weight factor

    # walk through all combinations of 0 and 1 on a length of a.ndim:
    #   0 is the shift with shift index floor(ctr[d]) for a dimension d
    #   1 the one for floor(ctr[d]) + 1
    comb_num = 2 ** a.ndim
    for comb_i in range(comb_num):
        comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)

        # add the weighted contribution for the shift corresponding to this combination
        cc = ctri + comb
        out += np.roll(np.roll(a, -cc[1], axis=1), -cc[0], axis=0) * ctrx[comb, range(a.ndim)].prod()

    return out


def shift_pixels(intensities, x_shifts, y_shifts):
    """
    Shift pixels in an intensity array according to x and y displacement fields.

    Parameters:
    -----------
    intensities : ndarray
        2D array containing the original intensity values
    x_shifts : ndarray
        2D array containing the x-direction shifts for each pixel
    y_shifts : ndarray
        2D array containing the y-direction shifts for each pixel

    Returns:
    --------
    ndarray
        2D array containing the shifted intensity values
    """
    # Create meshgrid of original pixel coordinates
    rows, cols = intensities.shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]

    # Calculate new positions for each pixel
    new_x = x_coords - x_shifts
    new_y = y_coords - y_shifts

    # Flatten arrays for interpolation
    points = np.column_stack((x_coords.flatten(), y_coords.flatten()))
    new_points = np.column_stack((new_x.flatten(), new_y.flatten()))

    # Perform interpolation to get intensity values at new positions
    shifted_intensities = griddata(
        points,
        intensities.flatten(),
        new_points,
        method='cubic',
        fill_value=0
    )

    # Reshape back to original dimensions
    return shifted_intensities.reshape(rows, cols)


def free_nf(w, l, z, pixsize=1.):
    """\
    Free-space propagation (near field) of the wavefield of a distance z.
    l is the wavelength.
    """
    sh = w.shape

    # Convert to pixel units.
    z = z / pixsize
    l = l / pixsize

    # Evaluate if aliasing could be a problem
    if min(sh) / np.sqrt(2.) < z * l:
        print
        "Warning: z > N/(sqrt(2)*lamda) = %.6g: this calculation could fail." % (min(sh) / (l * np.sqrt(2.)))
        print
        "(consider padding your array, or try a far field method)"

    q2 = np.sum((np.fft.ifftshift(
        np.indices(sh).astype(float) - np.reshape(np.array(sh) // 2, (len(sh),) + len(sh) * (1,)),
        range(1, len(sh) + 1)) * np.array([1. / sh[0], 1. / sh[1]]).reshape((2, 1, 1))) ** 2, axis=0)

    return np.fft.ifftn(np.fft.fftn(w) * np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - q2 * l ** 2) - 1)))


def savgol_2d_derivative(data, direction='x', window_length=5, polyorder=2):
    """
    Calculate the derivative of 2D data using a Savitzky-Golay filter in either x or y direction.

    Parameters:
    -----------
    data : ndarray
        2D input array of shape (m, n)
    direction : str
        Direction for derivative calculation: 'x' or 'y'
    window_length : int, optional
        Length of the filter window. Must be odd and greater than polyorder.
        Default is 5.
    polyorder : int, optional
        Order of the polynomial used to fit the samples. Must be less than
        window_length. Default is 2.

    Returns:
    --------
    ndarray
        2D array of same shape as input containing the directional derivative
    """
    # Input validation
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    if direction not in ['x', 'y']:
        raise ValueError("direction must be either 'x' or 'y'")

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")

    if window_length < polyorder + 1:
        raise ValueError("window_length must be greater than polyorder")

    # For y-direction, transpose the data, calculate derivative, then transpose back
    if direction == 'y':
        data = data.T

    # Calculate derivative
    derivative = np.zeros_like(data, dtype=float)

    for i in range(data.shape[0]):
        # Apply Savitzky-Golay filter with deriv=1 for first derivative
        derivative[i, :] = savgol_filter(
            data[i, :],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=1.0
        )

    # Transpose back if we were calculating y-direction derivative
    if direction == 'y':
        derivative = derivative.T

    return derivative


def bilateral_filter(image, sigma_spatial=2.0, sigma_intensity=50.0, window_size=5):
    """
    Apply bilateral filter to an image

    Parameters:
    -----------
    image : ndarray
        Input image (grayscale or color)
    sigma_spatial : float
        Standard deviation for spatial gaussian kernel
    sigma_intensity : float
        Standard deviation for intensity gaussian kernel
    window_size : int
        Size of the filter window (should be odd)

    Returns:
    --------
    ndarray
        Filtered image
    """
    # Ensure window size is odd
    window_size = max(3, window_size)
    if window_size % 2 == 0:
        window_size += 1

    # Pad the image to handle borders
    pad_size = window_size // 2
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

    # Create spatial gaussian kernel
    x, y = np.meshgrid(np.arange(window_size), np.arange(window_size))
    center = window_size // 2
    spatial_kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma_spatial ** 2))

    # Initialize output
    filtered = np.zeros_like(image, dtype=np.float32)

    # Apply filter
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            # Extract window
            window = padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]

            # Calculate intensity gaussian
            intensity_diff = window - padded[i, j]
            intensity_kernel = np.exp(-(intensity_diff ** 2) / (2 * sigma_intensity ** 2))

            # Combine kernels
            kernel = spatial_kernel * intensity_kernel
            kernel = kernel / np.sum(kernel)

            # Apply filter
            filtered[i - pad_size, j - pad_size] = np.sum(window * kernel)

    return filtered


from scipy import ndimage as ndi
import scipy
import matplotlib.pyplot as plt
import umpa3
import numpy as np

# Simulation of a sphere
dim = 512
sh = (dim, dim)
lam = .5e-10  # wavelength
z = 0.01  # propagation distance
psize = 2e-6  # pixel size
wavevec = 2 * np.pi / lam
blur_size = 1
period = 15


def create_TAI(shape, spot_spacing=16, sigma=1, vis=0.5, plot_visibility=False):
    y_shape = shape[0] + spot_spacing
    x_shape = shape[1] + spot_spacing

    xx, yy = np.indices((y_shape, x_shape))
    spots = np.zeros((y_shape, x_shape))

    # Calculate number of spots needed
    n_spots_x = x_shape // spot_spacing
    n_spots_y = y_shape // spot_spacing

    # First create spots with amplitude 1
    for i in range(n_spots_x):
        for j in range(n_spots_y):
            x_center = (i + 0.5) * spot_spacing
            y_center = (j + 0.5) * spot_spacing
            spots += np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2))
    spots *= 2
    spots -= 1
    B = 1
    spots = 1 + spots * vis
    spots = spots[spot_spacing // 2:-spot_spacing // 2, spot_spacing // 2:-spot_spacing // 2]

    I_max = np.max(spots)
    I_min = np.min(spots)
    achieved_visibility = (I_max - I_min) / (I_max + I_min)
    print('Achieved visibility:', round(achieved_visibility, 3))

    if plot_visibility:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot grating
        im1 = ax1.imshow(spots, cmap='gray', aspect='equal')
        ax1.set_title('Grating')
        plt.colorbar(im1, ax=ax1)

        # Add line plot through middle row of spots
        central_row = shape[0] // 2
        ax2.plot(np.arange(shape[1]), spots[central_row], 'b-')
        ax2.set_title('Intensity Profile (Central Row)')
        ax2.set_xlabel('Position (pixels)')
        ax2.set_ylabel('Intensity')
        ax2.grid(True)

        # Add vertical lines at spot centers for reference
        for i in range(n_spots_x):
            x_center = (i + 0.5) * spot_spacing
            ax3.axvline(x=x_center, color='r', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    return spots

def create_shift_positions(period, n_steps):
    step = period/n_steps * np.arange(n_steps)
    xx, yy = np.meshgrid(step, step)
    return np.column_stack((xx.ravel(), yy.ravel()))

xx, yy = np.indices(sh)
sphere_radius = 150
sphere = np.zeros_like(xx)
inside = (sphere_radius**2 - (xx-dim//2.)**2 - (yy-dim//2.)**2) >= 0
sphere[inside] = np.sqrt(sphere_radius**2 - (xx-dim//2.)**2 - (yy-dim//2.)**2)[inside]
sample = np.exp(-.2*np.pi*2j*sphere/sphere_radius)
#grid = np.exp(1j * 0.5 * np.pi * create_TAI(sh, spot_spacing=period, sigma=2, vis=0.3))
period = 15
z = 100e-2
grid = create_TAI(sh, spot_spacing=period, sigma=3, vis=.2)
reference = abs(free_nf(grid, lam, z, psize)) ** 2
pos = create_shift_positions(period, 3)
sref = [pshift(reference, p) for p in pos]

measurements = np.array([abs(free_nf(sample * pshift(grid, p), lam, z, psize)) ** 2 for p in pos])
result, status, iter = umpa3.run_py(np.ascontiguousarray(measurements).astype(np.float32),
                                    np.ascontiguousarray(sref).astype(np.float32), 5, False, 50)

dy, dx, T, err = result

from monash_processing.utils.ImageViewer import ImageViewer as imshow
import matplotlib
matplotlib.use('TkAgg', force=True)

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def correction_function(data, alpha, laplacian):
    return data * (1 + alpha * laplacian)

def calculate_mse(data):
    return np.mean(data**2)


def minimize_mse(measurements, laplacian):
    def objective(alpha):
        result = correction_function(measurements, alpha, laplacian)
        return calculate_mse(result)

    # Find optimal alpha
    result = minimize(objective, x0=1.0, method='Nelder-Mead')
    optimal_alpha = result.x[0]

    return optimal_alpha

from scipy.ndimage import gaussian_filter
imshow(T*(1+laplace_phi_filtered))

first_T = T
iterations = 100
from tqdm import tqdm
for i in tqdm(range(iterations)):
    # Apply bilateral filter
    #filtered_dx = bilateral_filter(dx, sigma_spatial=1/(i+1), sigma_intensity=0.1, window_size=5)
    #filtered_dy = bilateral_filter(dy, sigma_spatial=1/(i+1), sigma_intensity=0.1, window_size=5)

    # Calculate derivatives
    d_dx_filtered = savgol_2d_derivative(dx, direction='x', window_length=5, polyorder=3)
    d_dy_filtered = savgol_2d_derivative(dy, direction='y', window_length=5, polyorder=3)
    laplace_phi_filtered = d_dx_filtered + d_dy_filtered

    # Optimize TV for measurements
    #alpha = minimize_mse(T, laplace_phi_filtered)
    #print(alpha)
    measurements_corrected = correction_function(measurements, np.random.random() * 0.5 + 0.5, laplace_phi_filtered)

    result, status, iter = umpa3.run_py(np.ascontiguousarray(measurements_corrected).astype(np.float32),
                                        np.ascontiguousarray(sref).astype(np.float32), 1, False, 50)

    dy, dx, T, err = result
    imshow(T)

    print(f"Iteration {i + 1} - Minimum MSE value: {calculate_mse(T):.4f}")




