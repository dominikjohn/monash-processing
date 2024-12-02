import numpy as np
from scipy.interpolate import griddata


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


# Example usage:
# Create sample data
size = 100
x = np.linspace(-5, 5, size)
y = np.linspace(-5, 5, size)
X, Y = np.meshgrid(x, y)

# Create a sample intensity pattern (e.g., a Gaussian)
intensities = np.exp(-(X ** 2 + Y ** 2) / 2)

# Create sample displacement fields
x_shifts = 0.5 * np.cos(X / 2)  # Sample x displacement field
y_shifts = 0.5 * np.sin(Y / 2)  # Sample y displacement field

# Apply the shifts
shifted_intensities = shift_pixels(intensities, x_shifts, y_shifts)