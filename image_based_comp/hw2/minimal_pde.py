import numpy as np
import scipy.signal
import skimage.color

def nonlinearDiffusionFilter(image: np.ndarray,
                             iterations=10,
                             lamb=1.0,
                             tau=0.125,
                             image_seq=None):
    """
    Nonlinear (Perona–Malik) diffusion with Charbonnier-type conductance:
        g(|∇u|) = 1 / sqrt(1 + (|∇u|/lamb)^2)
    Explicit time stepping, zero-flux boundaries via padded arrays.
    """

    def computeDiffusivity(u: np.ndarray, lamb: float) -> np.ndarray:
        # If RGB, convert to gray
        if u.ndim == 3 and u.shape[2] > 1:
            u = skimage.color.rgb2gray(u)

        gradkernelx = 0.5 * np.array([[ 0.0, 0.0, 0.0],
                                      [-1.0, 0.0, 1.0],
                                      [ 0.0, 0.0, 0.0]])
        gradkernely = 0.5 * np.array([[ 0.0,-1.0, 0.0],
                                      [ 0.0, 0.0, 0.0],
                                      [ 0.0, 1.0, 0.0]])

        # IMPORTANT: mode='same'
        gradx = scipy.signal.convolve2d(u, gradkernelx, mode='same', boundary='symm')
        grady = scipy.signal.convolve2d(u, gradkernely, mode='same', boundary='symm')

        gradm2 = gradx*gradx + grady*grady
        g = 1.0 / np.sqrt(1.0 + gradm2 / (lamb*lamb))  # Charbonnier/PM variant
        return g

    def computeUpdate(u: np.ndarray, g: np.ndarray) -> np.ndarray:
        update = np.zeros_like(u, dtype=float)

        # Pad with zeros to block flux beyond boundary
        up = np.pad(u, pad_width=1, mode='constant')
        gp = np.pad(g, pad_width=1, mode='constant')

        # Loop (readable; you can vectorize later)
        for j in range(1, up.shape[0]-1):
            for i in range(1, up.shape[1]-1):
                g_pj = math.sqrt(gp[j, i+1] * gp[j, i])
                g_nj = math.sqrt(gp[j, i-1] * gp[j, i])
                g_ip = math.sqrt(gp[j+1, i] * gp[j, i])
                g_in = math.sqrt(gp[j-1, i] * gp[j, i])  # fixed index

                # If you keep zero padding, the boundary neighbors are already zeros.
                # These guards are optional; they ensure no flux at the very outer ring.
                if i == up.shape[1]-2: g_pj = 0.0
                if i == 1:             g_nj = 0.0
                if j == up.shape[0]-2: g_ip = 0.0
                if j == 1:             g_in = 0.0

                ux0 =  g_pj * (up[j, i+1] - up[j, i])
                ux1 = -g_nj * (up[j, i]   - up[j, i-1])
                uy0 =  g_ip * (up[j+1, i] - up[j, i])
                uy1 = -g_in * (up[j, i]   - up[j-1, i])

                update[j-1, i-1] = ux0 + ux1 + uy0 + uy1

        return update

    u = image.astype(float).copy()
    if u.ndim == 3 and u.shape[2] == 1:
        u = u[..., 0]

    if image_seq is not None:
        image_seq.append(u.copy())

    for k in range(iterations):
        g = computeDiffusivity(u, lamb)
        upd = computeUpdate(u, g)
        u = u + tau * upd
        if image_seq is not None:
            image_seq.append(u.copy())

    return u
