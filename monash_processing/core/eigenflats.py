import numpy as np
import cv2
import time
import scipy
import scipy.ndimage as nd
from monash_processing.utils.utils import Utils
from skimage.measure import block_reduce

class EigenflatManager:

    @staticmethod
    def eigenflats_PCA(flats, variance_threshold=0.99, max_components=None):
        """
        Compute eigenflats using PCA with intelligent component selection.

        Parameters:
        - flats: Input flat field images
        - variance_threshold: Keep enough components to explain this fraction of variance (default 0.99 or 99%)
        - max_components: Optional maximum number of components to keep

        Returns:
        - EFs: Selected eigenflats
        - MFs: Mean flats
        - n_components: Number of components selected
        - explained_var: Cumulative explained variance ratio
        """
        start = time.time()

        MFs = flats.mean(axis=0)
        A = flats - MFs
        A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])

        X = np.dot(A, A.T)
        # Use eigh for better numerical stability with symmetric matrices
        evi, vri = np.linalg.eigh(X)
        # Sort eigenvalues and vectors in descending order
        idx = np.argsort(evi)[::-1]
        ev, vr = evi[idx], vri[:, idx]

        # Calculate explained variance ratio
        explained_variance_ratio = ev / np.sum(ev)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Determine number of components to keep
        n_components = np.sum(cumulative_variance <= variance_threshold) + 1
        if max_components is not None:
            n_components = min(n_components, max_components)

        # Print variance explanation analysis
        print(f"\nVariance analysis:")
        print(f"Components needed for {variance_threshold * 100}% variance: {n_components}")
        print("\nCumulative variance explained by components:")
        for i in [1, 5, 10, n_components]:
            print(f"First {i:2d} components: {cumulative_variance[i - 1] * 100:.1f}%")

        EFs = np.dot(vr.T, A)
        # Normalize by eigenvalues
        EFs /= np.sqrt(np.abs(ev))[:, None]
        EFs = EFs.reshape(flats.shape)

        # Keep only the selected number of components
        EFs = EFs[:n_components]

        print(f'\nEigenflat generation time: {np.round(time.time() - start, 2)}s')
        print(f'Generated {n_components} components')

        return EFs, MFs, n_components, cumulative_variance[:n_components]

    @staticmethod
    def match_pca_to_proj(step, eigenflats, mean_flats, darkcurrent, area, projection):
        """
        worker function for loading data and generating FF image
        Can perform dead pixel correction

        """
        im = projection - darkcurrent
        im[im <= 0] = 1

        #im = Utils.perform_bad_pixel_correction(im, dead_pix_thl)

        #imc = cv2.GaussianBlur(im / mean_flats[step], (0, 0), 3)
        #m = nd.binary_erosion(np.abs(imc - 1) < 0.03, iterations=20)

        # use predefined background
        m = np.zeros_like(im, dtype=bool)
        area = np.s_[10:-10, -200:-10]
        m[area] = True

        # calculate weights of EFs using a lstsq min in the masked area
        sh = eigenflats[step, 0][m].shape
        ef = eigenflats[step][:, m].reshape(eigenflats.shape[1], sh[0])
        sm = (im[m] - mean_flats[step][m]).ravel()
        res = scipy.optimize.lsq_linear(np.swapaxes(ef, 0, 1), sm)


        losses = []

        def track_loss(xk):
            # Calculate residual: ||Ax - b||^2
            resid = np.linalg.norm(np.dot(np.swapaxes(ef, 0, 1), xk) - sm)**2
            losses.append(resid)


        bin_factor = 4
        proj_binned = block_reduce(im, (bin_factor, bin_factor), np.mean)
        eigenflats_binned = block_reduce(eigenflats[step], (1,bin_factor,bin_factor), np.mean)
        mean_flats_binned = block_reduce(mean_flats[step], (bin_factor, bin_factor), np.mean)
        binned_m = np.ones_like(proj_binned, dtype=bool)

        sh = eigenflats_binned[0][binned_m].shape
        ef = eigenflats_binned.reshape(eigenflats_binned.shape[0], sh[0])
        sm = (proj_binned - mean_flats_binned).ravel()
        res2 = scipy.optimize.lsq_linear(np.swapaxes(ef, 0, 1), sm)

        res = (res2['x'] + res1['x']) / 2

                # generate flat as weighted sum of EFs and MF
        mflat = np.sum(res['x'][:, None, None] * eigenflats[step], axis=0) + mean_flats[step]

        return im, mflat, res['x']