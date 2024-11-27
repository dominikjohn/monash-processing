import numpy as np
import cv2
import time

class EigenflatManager:

    def __init__(self, data_loader, flats):
        self.data_loader = data_loader
        self.flats = flats

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

        EFs = EFs[:10] # use only the first 10 eigenflats

        print('Gaussian-blurring components slightly')
        for i in range(EFs.shape[0]):
            EFs[i] = cv2.GaussianBlur(EFs[i], (0, 0), 0.5)

        print('\n Eigenflat generation time:', np.round(time.time() - start, 2), 's')

        return EFs, MFs