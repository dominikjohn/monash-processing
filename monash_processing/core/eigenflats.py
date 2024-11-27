import numpy as np
import cv2
import time

class EigenflatManager:

    @staticmethod
    def eigenflats_PCA(flats):

        start = time.time()

        MFs = flats.mean(axis=0)

        A = flats - MFs
        A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])

        X = np.dot(A, A.T)
        evi, vri = np.linalg.eig(X)
        idx = np.argsort(evi)[::-1]
        ev, vr = evi[idx], vri[:, idx]
        EFs = np.dot(vr.T, A)
        EFs /= np.sqrt(np.abs(ev))[:, None] # Normalize flat fields using the eigenvalues
        EFs = EFs.reshape(flats.shape)

        EFs = EFs[:15] # use only the first 15 eigenflats

        print('Gaussian-blurring components slightly')
        for i in range(EFs.shape[0]):
            EFs[i] = cv2.GaussianBlur(EFs[i], (0, 0), 0.5)

        print('\n Eigenflat generation time:', np.round(time.time() - start, 2), 's')

        return EFs, MFs

    @staticmethod
    def match_pca_to_proj(eigenflats, mean_flats, darkcurrent, area, projection):
        """
        worker function for loading data and generating FF image
        Can perform dead pixel correction

        """
        im = projection - darkcurrent
        im[im <= 0] = 1

        im = Utils.perform_bad_pixel_correction(im, dead_pix_thl)

        if not meaned_flats:
            if masking:
                # segment background, assuming transmission 1
                # slower but often better
                imc = cv2.GaussianBlur(im / MFs[i], (0, 0), 3)
                m = nd.binary_erosion(np.abs(imc - 1) < 0.03, iterations=20)

            else:
                # use predefined background
                m = np.zeros_like(im, dtype=bool)
                m[area] = 1

            # calculate weights of EFs using a lstsq min in the masked area
            sh = EFs[i, 0][m].shape
            ef = EFs[i][:, m].reshape(EFs.shape[1], sh[0])
            sm = (im[m] - MFs[i][m]).ravel()
            res = scipy.optimize.lsq_linear(np.swapaxes(ef, 0, 1), sm)

            # generate flat as weighted sum of EFs and MF
            mflat = np.sum(res['x'][:, None, None] * EFs[i], axis=0) + MFs[i]
        else:
            mflat = MFs[i]
            res = {'x': np.zeros(EFs[i].shape[0])}

        return i, im, mflat, res['x']