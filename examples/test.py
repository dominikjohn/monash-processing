import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib
from scipy.ndimage import gaussian_filter

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk


def generate_grid_with_spots():
    # Create a 512x512 dark grid
    grid = np.zeros((512, 512))

    # Generate positions for both positive and negative spots
    spots = []
    while len(spots) < 50:  # 8 positive + 8 negative spots
        x = randint(8, 503)  # 512 - 8 to ensure spot fits
        y = randint(8, 503)

        # Avoid center region and check for overlap with existing spots
        if abs(x - 256) > 50 or abs(y - 256) > 50:
            # Check distance from all existing spots
            valid_position = True
            for spot_x, spot_y, _ in spots:
                if np.sqrt((x - spot_x) ** 2 + (y - spot_y) ** 2) < 20:  # Minimum distance between spots
                    valid_position = False
                    break

            if valid_position:
                # First 8 spots are positive, next 8 are negative
                polarity = 1 if len(spots) < 8 else 1
                spots.append((x, y, polarity))


    # Place spots with Gaussian profile
    for x, y, polarity in spots:
        grid[y:y + 3, x:x + 3] += polarity * 1

    return grid, spots


def display_grid(grid):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray')
    plt.title(f'512x512 Grid with 8 Positive and 8 Negative Gaussian Spots')
    plt.colorbar()
    plt.show()


# Generate and display the grid
grid, spot_positions = generate_grid_with_spots()
display_grid(grid)

# Compute Fourier transform
fourier = np.fft.fft2(grid)
display_grid(np.real(fourier))