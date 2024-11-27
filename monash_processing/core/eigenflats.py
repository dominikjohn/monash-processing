import numpy as np
import cv2
import time

class EigenflatManager:

    def __init__(self, data_loader, flats):
        self.data_loader = data_loader
        self.flats = flats

    @staticmethod
    def eigenflats_PCA(flats):
        '''
        Performs a principle component analysis, to create eigenflatfield images.

        You might need some criterion to select the number of components necessary.
        E.g. look at the Eigenvalues and do a Scree plot.

        Parameters
        ----------
        flats : array
            array with all flat field images of the measurement, orderd as usual

        ncomp : int
            number of components to save. The fewer the faster the optimization
            afterwards

        eigenvalue_return : bool, Default = False
            return of the eigenvalues

        Returns
        --------
        EFs : array
            Eigenflatfields, with the order (phasestep, component, y, x)

        MFs : array
            Mean Flat Field image (phasestep, y, x)

        ev : array
            eigenvalues, only returned if eigenvalue_return = True

        '''
        start = time.time()

        MFs = self.flats.mean(axis=0)

        A = flats - MFs
        A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])

        X = np.dot(A, A.T)
        evi, vri = np.linalg.eig(X)
        idx = np.argsort(evi)[::-1]
        ev, vr = evi[idx], vri[:, idx]
        EFs = np.dot(vr.T, A)
        EFs /= np.sqrt(np.abs(ev))[:, None] # Normalize flat fields using the eigenvalues
        EFs = EFs.reshape(flats.shape)

        print('Gaussian-blurring components slightly')
        for i in range(EFs.shape[0]):
            for j in range(EFs.shape[1]):
                EFs[i, j] = cv2.GaussianBlur(EFs[i, j], (0, 0), 0.5)

        print('\n Eigenflat generation time:', np.round(time.time() - start, 2), 's')

        return EFs, MFs