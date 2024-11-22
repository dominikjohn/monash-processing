import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


class ImageViewer:
    def __init__(self, image, title=None):
        """
        Create and display an interactive image viewer.

        Args:
            image (np.ndarray): 2D array to display
            title (str, optional): Title for the plot
        """
        self.image = image

        # Create figure and axes
        self.fig, (self.ax_img, self.ax_slider_min, self.ax_slider_max) = plt.subplots(
            3, 1, gridspec_kw={'height_ratios': [6, 1, 1]}, figsize=(8, 10))

        # Calculate initial values
        self.vmin_init = np.percentile(image, 1)
        self.vmax_init = np.percentile(image, 99)
        self.abs_min = np.min(image)
        self.abs_max = np.max(image)

        # Display image
        self.img_display = self.ax_img.imshow(image, vmin=self.vmin_init, vmax=self.vmax_init)
        if title:
            self.ax_img.set_title(title)

        # Add colorbar
        self.colorbar = plt.colorbar(self.img_display, ax=self.ax_img)

        # Create sliders
        self.slider_min = Slider(self.ax_slider_min, 'Min', self.abs_min, self.abs_max,
                                 valinit=self.vmin_init, orientation='horizontal')
        self.slider_max = Slider(self.ax_slider_max, 'Max', self.abs_min, self.abs_max,
                                 valinit=self.vmax_init, orientation='horizontal')

        # Register update function
        self.slider_min.on_changed(self._update)
        self.slider_max.on_changed(self._update)

        plt.tight_layout()
        plt.show()

    def _update(self, _):
        """Update the image display based on slider values."""
        vmin = self.slider_min.val
        vmax = self.slider_max.val
        if vmin < vmax:  # Only update if min is less than max
            self.img_display.set_clim(vmin, vmax)
            self.fig.canvas.draw_idle()