import numpy as np

def match_speckles(Isample, Iref, Nw, step=1, max_shift=4, df=True, printout=True):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) using a given window.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param Nw: 2*Nw + 1 is the width of the window.
    :param step: perform the analysis on every other _step_ pixels in both directions (default 1)
    :param max_shift: Do not allow shifts larger than this number of pixels (default 4)
    :param df: Compute dark field (default True)

    Return T, dx, dy, df, f
    """

    Ish = Isample[0].shape

    # Create the window
    w = np.multiply.outer(np.hamming(2*Nw+1), np.hamming(2*Nw+1))
    w /= w.sum()

    NR = len(Isample)

    S2 = sum(I**2 for I in Isample)
    R2 = sum(I**2 for I in Iref)
    if df:
        S1 = sum(I for I in Isample)
        R1 = sum(I for I in Iref)
        Im = R1.mean()/NR

    L1 = cc(S2, w)
    L3 = cc(R2, w)
    if df:
        L2 = Im * Im * NR
        L4 = Im * cc(S1, w)
        L6 = Im * cc(R1, w)
    # (We need a loop for L5)

    # 2*Ns + 1 is the width of the window explored to find the best fit.
    Ns = max_shift

    ROIx = np.arange(Ns+Nw, Ish[0]-Ns-Nw-1, step)
    ROIy = np.arange(Ns+Nw, Ish[1]-Ns-Nw-1, step)

    # The final images will have this size
    sh = (len(ROIy), len(ROIx))
    tx = np.zeros(sh)
    ty = np.zeros(sh)
    tr = np.zeros(sh)
    do = np.zeros(sh)
    MD = np.zeros(sh)

    # Loop through all positions
    for xi, i in enumerate(ROIy):
        for xj, j in enumerate(ROIx):
            # Define local values of L1, L2, ...
            t1 = L1[i, j]
            t3 = L3[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
            if df:
                t2 = L2
                t4 = L4[i, j]
                t6 = L6[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
            else:
                t2 = 0.
                t4 = 0.
                t6 = 0.

            # Now we can compute t5 (local L5)
            t5 = np.zeros((2*Ns+1, 2*Ns+1))
            for k in range(NR):
                t5 += cc(Iref[k][(i-Nw-Ns):(i+Nw+Ns+1), (j-Nw-Ns):(j+Nw+Ns+1)],
                         w * Isample[k][(i-Nw):(i+Nw+1), (j-Nw):(j+Nw+1)], mode='valid')

            # Compute K and beta
            if df:
                K = (t2*t5 - t4*t6)/(t2*t3 - t6**2)
                beta = (t3*t4 - t5*t6)/(t2*t3 - t6**2)
            else:
                K = t5/t3
                beta = 0.

            # Compute v and a
            a = beta + K
            v = K/a

            # Construct D
            D = t1 + (beta**2)*t2 + (K**2)*t3 - 2*beta*t4 - 2*K*t5 + 2*beta*K*t6

            # Find subpixel optimum for tx an ty
            sy, sx = sub_pix_min(D)

            # We should re-evaluate the other values with sub-pixel precision but here we just round
            # We also need to clip because "sub_pix_min" can return the position of the minimum outside of the bounds...
            isy = np.clip(int(np.round(sy)), 0, 2*Ns)
            isx = np.clip(int(np.round(sx)), 0, 2*Ns)

            # store everything
            ty[xi, xj] = sy - Ns
            tx[xi, xj] = sx - Ns
            tr[xi, xj] = a[isy, isx]
            do[xi, xj] = v[isy, isx]
            MD[xi, xj] = D[isy, isx]

    return {'T': tr, 'dx': tx, 'dy': ty, 'df': do, 'f': MD}

