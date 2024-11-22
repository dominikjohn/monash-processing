import numpy as np
from scipy import ndimage

class BadPixelMask:

    @staticmethod
    def detect_bad_pixels(image, nsigma=5, neighborhood_size=5):
        """
        Detect bad pixels using median filtering and statistical thresholding.

        Parameters:
        -----------
        image : 2D numpy array
            Input image
        nsigma : float
            Number of standard deviations for thresholding
        neighborhood_size : int
            Size of the neighborhood for median filtering

        Returns:
        --------
        bad_pixel_mask : 2D boolean array
            True where bad pixels are detected
        """
        # Calculate median filtered image
        median_filtered = ndimage.median_filter(image, size=neighborhood_size)

        # Calculate deviation from median
        deviation = np.abs(image - median_filtered)

        # Calculate robust statistics
        mad = np.median(np.abs(deviation - np.median(deviation)))
        threshold = nsigma * mad / 0.6745  # Convert MAD to sigma equivalent

        # Create bad pixel mask
        bad_pixel_mask = deviation > threshold

        return bad_pixel_mask

    @staticmethod
    def correct_bad_pixels(image, bad_pixel_mask=None, nsigma=5, neighborhood_size=5,
                           correction_method='median'):
        """
        Correct bad pixels in an image using various methods.

        Parameters:
        -----------
        image : 2D numpy array
            Input image to correct
        bad_pixel_mask : 2D boolean array or None
            Pre-defined bad pixel mask. If None, will be automatically detected
        nsigma : float
            Number of standard deviations for thresholding if auto-detecting bad pixels
        neighborhood_size : int
            Size of the neighborhood for filtering
        correction_method : str
            Method to use for correction ('median', 'mean', or 'interpolate')

        Returns:
        --------
        corrected_image : 2D numpy array
            Image with bad pixels corrected
        bad_pixel_mask : 2D boolean array
            Mask showing which pixels were corrected
        """
        # Make a copy of the input image
        corrected_image = image.copy()

        # Detect bad pixels if mask not provided
        if bad_pixel_mask is None:
            bad_pixel_mask = BadPixelMask.detect_bad_pixels(image, nsigma, neighborhood_size)

        # Create a larger neighborhood for correction
        correction_size = neighborhood_size + 2

        if correction_method == 'median':
            # Use median filtering
            corrected_values = ndimage.median_filter(image, size=correction_size)
        elif correction_method == 'mean':
            # Use mean filtering
            corrected_values = ndimage.uniform_filter(image, size=correction_size)
        elif correction_method == 'interpolate':
            # Use interpolation (more sophisticated method)
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            good_coords = np.column_stack((y[~bad_pixel_mask], x[~bad_pixel_mask]))
            good_values = image[~bad_pixel_mask]
            bad_coords = np.column_stack((y[bad_pixel_mask], x[bad_pixel_mask]))

            # Use nearest neighbor interpolation for speed
            from scipy.interpolate import NearestNDInterpolator
            interpolator = NearestNDInterpolator(good_coords, good_values)
            corrected_values = image.copy()
            corrected_values[bad_pixel_mask] = interpolator(bad_coords)
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")

        # Apply correction only to bad pixels
        corrected_image[bad_pixel_mask] = corrected_values[bad_pixel_mask]

        # Add edge handling
        edge_mask = np.zeros_like(bad_pixel_mask)
        edge_mask[0:neighborhood_size] = True
        edge_mask[-neighborhood_size:] = True
        edge_mask[:, 0:neighborhood_size] = True
        edge_mask[:, -neighborhood_size:] = True

        # Use simple median filtering for edges
        if np.any(bad_pixel_mask & edge_mask):
            edge_correction = ndimage.median_filter(image, size=3)
            corrected_image[bad_pixel_mask & edge_mask] = edge_correction[bad_pixel_mask & edge_mask]

        return corrected_image, bad_pixel_mask

    @staticmethod
    def analyze_correction(original_image, corrected_image, bad_pixel_mask):
        """
        Analyze the results of bad pixel correction.

        Parameters:
        -----------
        original_image : 2D numpy array
            Original image before correction
        corrected_image : 2D numpy array
            Image after correction
        bad_pixel_mask : 2D boolean array
            Mask showing which pixels were corrected

        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        results = {
            'num_bad_pixels': np.sum(bad_pixel_mask),
            'percentage_bad': 100 * np.sum(bad_pixel_mask) / bad_pixel_mask.size,
            'mean_difference': np.mean(np.abs(original_image - corrected_image)[bad_pixel_mask]),
            'max_difference': np.max(np.abs(original_image - corrected_image)[bad_pixel_mask])
        }

        return results