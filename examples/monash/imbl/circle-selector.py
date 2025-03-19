import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import glob
import os

# Base path to the TIFF files
base_path = '/data/imbl/23081/output-during-beamtime/Day3/Dominik_KI_salts_0p75m_30keV_0p16s/recon_phase/binned16/'
file_pattern = 'recon_cs00885_idx_*_binned.tiff'


# Interactive circle selection
def select_circle(img):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title("Click center, then edge to define circle")

    points = plt.ginput(2)  # Get two mouse clicks
    center = points[0]
    edge = points[1]
    radius = np.sqrt((center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2)

    # Draw the circle
    circle = plt.Circle(center, radius, fill=False, edgecolor='r')
    plt.gca().add_patch(circle)
    plt.title(f"Center: {center}, Radius: {radius:.1f}")
    plt.draw()
    plt.pause(1)

    return (center[0], center[1], radius)


# Create circular mask
def create_circle_mask(img_shape, center, radius):
    y, x = np.ogrid[:img_shape[0], :img_shape[1]]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


# Main script
print("Simple Pipe Analysis")

# Get list of files
tiff_files = sorted(glob.glob(os.path.join(base_path, file_pattern)))
if not tiff_files:
    print(f"No files found matching pattern: {os.path.join(base_path, file_pattern)}")
    exit()

print(f"Found {len(tiff_files)} TIFF files")
for i, f in enumerate(tiff_files[:10]):  # Show first 10 files
    print(f"{i}: {os.path.basename(f)}")
print("...")
for i, f in enumerate(tiff_files[-10:], start=len(tiff_files) - 10):  # Show last 10 files
    print(f"{i}: {os.path.basename(f)}")

# Files to process
files_to_analyze = tiff_files
num_files = len(files_to_analyze)
print(f"Analyzing {num_files} files")

# Load first and last images
first_img = io.imread(files_to_analyze[0])
last_img = io.imread(files_to_analyze[-1])

# Results storage
pipe_results = {}

# Process each pipe
for pipe_num in range(4):
    print(f"\nPipe {pipe_num + 1}:")

    # Select circles on first and last slices
    print("Select circle on first slice:")
    first_circle = select_circle(first_img)

    print("Select circle on last slice:")
    last_circle = select_circle(last_img)

    # Collect all values
    all_values = []

    # Process each file
    for i, file_path in enumerate(files_to_analyze):
        # Calculate interpolation factor
        factor = i / (num_files - 1) if num_files > 1 else 0

        # Interpolate circle parameters
        center_x = first_circle[0] + factor * (last_circle[0] - first_circle[0])
        center_y = first_circle[1] + factor * (last_circle[1] - first_circle[1])
        radius = first_circle[2] + factor * (last_circle[2] - first_circle[2])

        # Load image
        img = io.imread(file_path)

        # Create mask
        mask = create_circle_mask(img.shape, (center_x, center_y), radius)

        # Get values inside mask
        values = img[mask]
        all_values.extend(values)

        # Save visualization for first, middle and last image
        if i == 0 or i == num_files // 2 or i == num_files - 1:
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            circle = plt.Circle((center_x, center_y), radius, fill=False, edgecolor='r')
            plt.gca().add_patch(circle)
            plt.title(f"Pipe {pipe_num + 1} - Image {i + 1}/{num_files}")
            os.makedirs("pipe_analysis", exist_ok=True)
            plt.savefig(f"pipe_analysis/pipe{pipe_num + 1}_slice{i}.png")
            plt.close()

    # Calculate statistics
    values_array = np.array(all_values)
    stats = {
        'mean': np.mean(values_array),
        'std': np.std(values_array),
        'min': np.min(values_array),
        'max': np.max(values_array),
        'count': len(values_array)
    }
    pipe_results[f"Pipe_{pipe_num + 1}"] = stats

    print(f"Results for Pipe {pipe_num + 1}:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  StdDev: {stats['std']:.2f}")
    print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"  Pixels analyzed: {stats['count']}")

# Print summary
print("\nSummary:")
print("-" * 50)
for pipe_name, stats in pipe_results.items():
    print(f"{pipe_name}: {stats['mean']:.2f} ± {stats['std']:.2f}")