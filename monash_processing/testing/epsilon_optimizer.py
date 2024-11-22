import numpy as np
from numpy import fft
from scipy.optimize import minimize_scalar


def calculate_phi(epsilon, mirror_I, mdx, mdy, k, l, constant, wavevec, delta_mu):
    """
    Calculate phi for a given epsilon value.
    """
    epsilon_sq = epsilon ** 2
    phi_coarse = np.log(mirror_I) * epsilon * wavevec * delta_mu

    # Calculate Phi_filtered
    numerator = 1j * k * fft.fft2(mdx) + 1j * l * fft.fft2(mdy) - epsilon_sq * fft.fft2(phi_coarse)
    denominator = epsilon_sq + k ** 2 + l ** 2

    return -np.real(fft.ifft2(numerator / denominator))[dx.shape[0]:, :dy.shape[1]] * constant


def objective_function(epsilon, mirror_I, mdx, mdy, k, l, background, constant, wavevec, delta_mu):
    """
    Objective function to minimize: mean absolute value in background + pi/2
    """
    try:
        phi = calculate_phi(epsilon, mirror_I, mdx, mdy, k, l, constant, wavevec, delta_mu)
        mean_abs = np.mean(phi[background]) + np.pi / 2
        return np.abs(mean_abs)  # We want to minimize the absolute value
    except:
        return np.inf  # Return infinity if calculation fails


def optimize_epsilon(mirror_I, mdx, mdy, k, l, background, constant, wavevec, delta_mu, verbose=True):
    """
    Optimize epsilon using scipy's minimize_scalar.
    """
    # Define the bounds for optimization
    bounds = (1e-10, 1e-8)

    # Wrapper function for the objective
    def objective_wrapper(x):
        return objective_function(x, mirror_I, mdx, mdy, k, l, background,
                                  constant, wavevec, delta_mu)

    # Run optimization
    if verbose:
        print("Starting optimization...")

    result = minimize_scalar(
        objective_wrapper,
        bounds=bounds,
        method='bounded',
        options={'xatol': 1e-12}  # Set very small tolerance for precise optimization
    )

    if verbose:
        print(f"\nOptimization completed:")
        print(f"Best epsilon found: {result.x:.2e}")
        print(f"Final objective value: {result.fun:.2e}")
        print(f"Optimization success: {result.success}")
        print(f"Number of iterations: {result.nfev}")

    # Calculate final phi with optimal epsilon
    if result.success:
        final_phi = calculate_phi(result.x, mirror_I, mdx, mdy, k, l,
                                  constant, wavevec, delta_mu)
        return result.x, final_phi
    else:
        print("Optimization failed!")
        return None, None


# Run the optimization
best_epsilon, best_phi = optimize_epsilon(
    mirror_I=mirror_I,
    mdx=mdx,
    mdy=mdy,
    k=k,
    l=l,
    background=background,
    constant=constant,
    wavevec=wavevec,
    delta_mu=delta_mu
)

if best_phi is not None:
    ImageViewerPhi(best_phi)