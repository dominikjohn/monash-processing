import numpy as np
import cv2
import rasterio.fill as rs
import scipy.constants
from monash_processing.core.data_loader import DataLoader
from scipy import fft
from monash_processing.postprocessing.bad_pixel_cor import BadPixelMask


class PhaseIntegrator:

    def __init__(self, energy, prop_distance, pixel_size, area_left, area_right, data_loader: DataLoader):
        self.energy = energy
        self.prop_distance = prop_distance
        self.pixel_size = pixel_size
        self.wavevec = 2 * np.pi * self.energy / (
                scipy.constants.physical_constants['Planck constant in eV s'][0] * scipy.constants.c)
        self.data_loader = data_loader
        self.area_left = area_left
        self.area_right = area_right

    def integrate_single(self, projection_i):

        # Load dx, dy, f
        dx = self.data_loader.load_processed_projection(projection_i, 'dx')
        dy = self.data_loader.load_processed_projection(projection_i, 'dy')

        # Create a mask for the ramp correction based on the previous user input
        mask = np.zeros_like(dx, dtype=bool)
        mask[self.area_left] = True
        mask[self.area_right] = True

        dx = np.clip(BadPixelMask.correct_bad_pixels(dx), -8, 8)
        dy = np.clip(BadPixelMask.correct_bad_pixels(dy), -8, 8)

        dx -= PhaseIntegrator.img_poly_fit(dx, order=1, mask=mask)
        dy -= PhaseIntegrator.img_poly_fit(dy, order=1, mask=mask)

        mdx = PhaseIntegrator.antisym_mirror_im(dx, 'dx')
        mdy = PhaseIntegrator.antisym_mirror_im(dy, 'dy')

        k = fft.fftfreq(mdx.shape[1])
        l = fft.fftfreq(mdy.shape[0])
        k[k == 0] = 1e-10
        l[l == 0] = 1e-10
        k, l = np.meshgrid(k, l)

        ft = fft.fft2(mdx + 1j * mdy, workers=2)
        phi_raw = fft.ifft2(ft / ((2 * np.pi * 1j) * (k + 1j * l)), workers=2)
        phi_raw = np.real(phi_raw)[dx.shape[0]:, dy.shape[1]:]

        phi_corr = phi_raw * (self.wavevec / self.prop_distance) * (self.pixel_size ** 2)

        p_phi_corr = self._img_poly_fit(phi_corr, order=1, mask=mask)
        phi_corr -= p_phi_corr

        if np.percentile(phi_corr, 99) > 1E3:
            print(f'Integration of projection {projection_i} failed!')
            phi_corr = np.zeros_like(phi_corr)

        self.data_loader.save_tiff('phi', projection_i, phi_corr)

    @staticmethod
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

    @staticmethod
    def sym_mirror_im(im, mode='reflect'):
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
        return m_im

    @staticmethod
    def cleanup_rio(image, f, thl):
        '''
        Corrects differential phase image for wrapping/bad UMPA fits.
        Requires rasterio to be installed.
        A lot faster than standard iterative cleanup with similar results.

        Creates bad pixel max out of df or f image and detected edges.
        Mask is iteratively closed. Selects extremum of input and corrected image.
        Multiplies corrected image edges by factor to account for loss of edges.

        Parameters
        ----------
        image : 2D-array
            Image to be corrected

        f : 2D-array
            UMPA residuum f

        thl : float
            THL for f image

        factor : float, Default=1
            Factor for improving the edges, needs to be determined empirically....

        axis : int, Default=1
            Selects which differential phase is given, 1 for dx, 0 for dy

        sigma : float, Default=4
            Selects sigma for 1D gauss blurring of mask for edge correction


        Returns
        --------
        rio_im : 2D-array
            Corrected image.
        '''

        if f is None:
            error_mask = np.zeros_like(image)
        else:
            error_mask = f > thl

        # Gaussian Mask: Masks out areas with strong phase jumps
        gauss = cv2.GaussianBlur(image, (0, 0), 8)
        gauss_mask = np.where(np.abs(image - gauss) < 1.2, 0, 1)
        gauss_mask.astype(bool)

        ######## Merge Masks
        merged_mask = error_mask + gauss_mask
        opened_mask = cv2.morphologyEx(merged_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(
            bool)

        rio_im = rs.fillnodata(image.copy(), np.invert(opened_mask), smoothing_iterations=0)
        dilated_mask = cv2.dilate(opened_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5).astype(bool)

        rio_imf = cv2.GaussianBlur(rio_im, (0, 0), 2)
        rio_im[dilated_mask] = rio_imf[dilated_mask]

        return rio_im

    @staticmethod
    def img_poly_fit(a, order=1, mask=None):
        """
        Returns a best fit of a to a 2D polynomial of given order, using only values in the mask.

        .. note::
            Pixel units.

        Parameters
        ----------
        a : 2D-numpy-array
            Array with values for fitting.

        order : int, Default=1, optional
            Order of fit polynomial.

        mask : bool, Default=None, optional
            Uses in the fit only the elements of ``a`` that have a True mask value (None means use all array). Same shape as ``a``.

        Returns
        --------
        2D-numpy-array
            Values of fitted polynomial.

        """

        sh = a.shape
        # 2D fit
        i0, i1 = np.indices(sh)
        i0f = i0.ravel()
        i1f = i1.ravel()
        af = a.ravel()

        if mask is not None:
            mf = mask.ravel()
            i0f = i0f[mf]
            i1f = i1f[mf]
            af = af[mf]

        A = np.vstack([i0f ** (i) * i1f ** (n - i) for n in range(order + 1) for i in range(n + 1)]).T
        r = np.linalg.lstsq(A, af, rcond=None)
        p = r[0]
        if mask is not None:
            i0f = i0.ravel()
            i1f = i1.ravel()
            A = np.vstack([i0f ** (i) * i1f ** (n - i) for n in range(order + 1) for i in range(n + 1)]).T

        return np.dot(A, p).reshape(sh)
