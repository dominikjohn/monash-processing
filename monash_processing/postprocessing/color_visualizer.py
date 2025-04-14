
class ColorVisualizer:
    def plot_transmittance_curve(self, wavelengths, extinction_coefficients, concentration):
        """Plot the transmittance curve for different thicknesses."""
        thicknesses = np.linspace(0, 500, 100)
        avg_transmittances = []

        for thickness in thicknesses:
            result = self.calculate_transmitted_spectrum(
                wavelengths,
                extinction_coefficients,
                thickness_um=thickness,
                concentration=concentration,
                light_color=6500
            )
            avg_transmittance = result['avg_transmittance']
            avg_transmittances.append(avg_transmittance)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thicknesses, avg_transmittances, 'b-', linewidth=2)
        ax.set_xlabel('Thickness (µm)')
        ax.set_ylabel('Average Transmittance')
        ax.set_title('Haematoxylin Light Transmission vs. Thickness')
        ax.grid(True)
        ax.set_ylim(0, 1)

        # Add reference line at 50% transmittance
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        ax.text(thicknesses[-1] * 0.8, 0.52, 'T = 0.5', color='r')

        # Add 10% transmittance line
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7)
        ax.text(thicknesses[-1] * 0.8, 0.12, 'T = 0.1', color='r')

        return fig

    def display_data(self, wavelengths, extinction_coefficients, concentration=10e-4):
        plt.figure()
        plt.plot(wavelengths, extinction_coefficients, label='Haematoxylin')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Extinction Coefficient')
        plt.title('Haematoxylin Absorption Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()
        fig1 = self.visualize_absorption_simple(wavelengths, extinction_coefficients, concentration=concentration)
        plt.show()
        fig2 = self.plot_transmittance_curve(wavelengths, extinction_coefficients, concentration=concentration)
        plt.show()

    @staticmethod
    def plot_slice(data, slice_idx, pixel_size,
                   cmap='grey',
                   title=None,
                   vmin=None,
                   vmax=None,
                   figsize=(10, 8),
                   fontsize=16,
                   percent=False,
                   colorbar_position='right'):  # New parameter for colorbar position

        # Set the font size globally
        plt.rcParams.update({'font.size': fontsize})

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Get the slice
        slice_data = data[slice_idx] if len(data.shape) == 3 else data

        if percent:
            # Plot the image
            im = ax.imshow(slice_data * 100,
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax)
        else:
            # Plot the image
            im = ax.imshow(slice_data,
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax)

        # Add scalebar
        scalebar = ScaleBar(pixel_size,  # meters per pixel
                            "m",  # meter unit
                            length_fraction=.2,
                            color='white',
                            box_alpha=0,
                            location='lower right',
                            font_properties={'size': fontsize})
        ax.add_artist(scalebar)

        # Add colorbar with matching height and title
        divider = make_axes_locatable(ax)

        # Position the colorbar according to the parameter
        if colorbar_position.lower() == 'left':
            cax = divider.append_axes("left", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            # For left position, we need to adjust the orientation of ticks
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')
        else:  # Default to right
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)

        cbar.set_label(f'{title}', size=fontsize, labelpad=15)
        cbar.ax.tick_params(labelsize=fontsize)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        return fig, ax
