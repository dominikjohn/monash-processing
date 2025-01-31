import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from skimage.draw import line


def paganin_filter(image, pixel_size, dist, wavelength, delta_beta_ratio):
    # Get image dimensions
    ny, nx = image.shape

    # Calculate frequencies using fftfreq
    delta_x = pixel_size / (2 * np.pi)
    kx = np.fft.fftfreq(nx, d=delta_x)
    ky = np.fft.fftfreq(ny, d=delta_x)

    # Create 2D frequency grid
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_squared = kx_grid ** 2 + ky_grid ** 2

    # Create Paganin filter with corrected formula
    # Since we're using delta_beta_ratio = delta/beta, need to multiply by 1/(4π)
    #denom = 1 + dist * wavelength * (delta_beta_ratio / (4 * np.pi)) * k_squared
    denom = 1 + dist * wavelength * delta_beta_ratio *  k_squared
    paganin_filter = 1 / denom

    # Apply filter in Fourier space
    image_fft = np.fft.fft2(image)
    filtered_fft = image_fft * paganin_filter
    filtered_image = np.real(np.fft.ifft2(filtered_fft))

    return filtered_image

def extract_line_profile(image, start_point, end_point):
    # Get points along the line
    rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])
    # Extract intensity values along the line
    return image[rr, cc]


# Parameters
pixel_size = 1.44e-6
dist = 0.155
wavelength = 0.05e-9
gamma_values = np.arange(100, 2500, 100)  # 10 to 200 in steps of 10

# Load image
#test3 = np.load('test3.npy')  # or however your image is stored

# Define line profile coordinates
#start_point = (768, 1624) # coordinates for pvc
#end_point = (842, 1656) #

#start_point = (1658, 1144)
#end_point = (1689, 1193)

start_point = (2582, 2513)
end_point = (2631, 2592)

#1473,1719| 1424, 1660
# Process each gamma value and extract profiles
profiles = []
for gamma in gamma_values:
    # Apply Paganin filter
    filtered_image = paganin_filter(test6, pixel_size, dist, wavelength, gamma)
    # Extract line profile
    profile = extract_line_profile(filtered_image, start_point, end_point)
    profiles.append(profile)

# Plot results
plt.figure(figsize=(12, 8))
for i, (gamma, profile) in enumerate(zip(gamma_values, profiles)):
    plt.plot(profile, label=f'γ = {gamma}', alpha=0.7)

plt.xlabel('Position along line')
plt.ylabel('Intensity')
plt.title('Line Profiles for Different Gamma Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def error_func(x, a, b, c, d):
    """
    Error function of form: a * erf((x-b)/c) + d
    a: amplitude
    b: center position
    c: width parameter
    d: vertical offset
    """
    return a * erf((x - b) / c) + d


# Process each gamma value
fit_qualities = []
fit_params = []
x = np.arange(len(profiles[0]))

plt.figure(figsize=(15, 10))

for gamma, profile in zip(gamma_values, profiles):
    try:
        # Normalize profile to [0,1] range for better fitting
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

        # Initial parameter guesses
        p0 = [
            0.5,  # amplitude guess
            len(profile_norm) / 2,  # center guess
            len(profile_norm) / 10,  # width guess
            np.mean(profile_norm)  # offset guess
        ]

        # Fit error function
        popt, pcov = curve_fit(error_func, x, profile_norm, p0=p0)

        # Calculate fit quality (R-squared)
        residuals = profile_norm - error_func(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((profile_norm - np.mean(profile_norm)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_qualities.append((gamma, r_squared))
        fit_params.append((gamma, popt))

        # Plot data and fit
        plt.plot(x, profile_norm, 'o', alpha=0.3, markersize=2, label=f'Data γ={gamma}')
        plt.plot(x, error_func(x, *popt), '--', alpha=0.7,
                 label=f'Fit γ={gamma}, R²={r_squared:.3f}')

    except RuntimeError as e:
        print(f"Fitting failed for gamma={gamma}: {str(e)}")
        continue

# Find best fit
best_gamma, best_r_squared = max(fit_qualities, key=lambda x: x[1])
print(f"\nBest fit achieved with γ = {best_gamma}")
print(f"R-squared value: {best_r_squared:.4f}")

# Get parameters for best fit
best_params = next(params[1] for params in fit_params if params[0] == best_gamma)
print("\nBest fit parameters:")
print(f"Amplitude (a): {best_params[0]:.4f}")
print(f"Center (b): {best_params[1]:.4f}")
print(f"Width (c): {best_params[2]:.4f}")
print(f"Offset (d): {best_params[3]:.4f}")

plt.xlabel('Position along line')
plt.ylabel('Normalized Intensity')
plt.title('Error Function Fits for Different Gamma Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()


#theoretical_gamma = 147.79
#experimental_gamma = 47
theoretical_gamma = 2232
experimental_gamma = 450

sigma = np.sqrt((theoretical_gamma-experimental_gamma)*wavelength*dist/(8*np.pi))