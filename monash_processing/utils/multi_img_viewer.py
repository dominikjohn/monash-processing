import matplotlib.pyplot as plt
import numpy as np


class SideBySideViewer:
    def __init__(self, images, titles=None, figsize=(15, 6), vmin=0, vmax=np.pi):
        """
        Create a viewer showing multiple images side by side.

        Args:
            images (list): List of 2D numpy arrays to display
            titles (list): List of strings for image titles
            figsize (tuple): Figure size (width, height)
            vmin (float): Minimum value for color scaling
            vmax (float): Maximum value for color scaling
        """
        self.images = images
        self.n_images = len(images)

        # Create figure
        self.fig, self.axes = plt.subplots(1, self.n_images, figsize=figsize)

        # Handle single image case
        if self.n_images == 1:
            self.axes = [self.axes]

        # Create all image subplots
        self.img_displays = []
        for i in range(self.n_images):
            # Display image
            img_display = self.axes[i].imshow(
                images[i],
                vmin=vmin,
                vmax=vmax,
                cmap='gray'
            )
            self.img_displays.append(img_display)

            # Set title
            if titles and i < len(titles):
                self.axes[i].set_title(titles[i])
            else:
                self.axes[i].set_title(f'Image {i + 1}')

            # Add colorbar
            plt.colorbar(img_display, ax=self.axes[i])

        plt.tight_layout()

    def show(self):
        """Display the viewer."""
        plt.show()