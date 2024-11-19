import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
import numpy as np


def view_images_tk(data, cmap='gray', title=None):
    """
    Enhanced Tk/matplotlib image viewer with contrast controls
    """
    root = tk.Tk()
    root.title(title or "Image Viewer")

    # Create the figure with gridspec for better layout
    fig = Figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 20)  # 20 columns for fine control
    ax = fig.add_subplot(gs[0, :19])  # Main plot takes 19 columns
    cax = fig.add_subplot(gs[0, 19])  # Colorbar takes 1 column

    # Variables for contrast control
    vmin = tk.DoubleVar(value=data.min())
    vmax = tk.DoubleVar(value=data.max())
    current_idx = tk.IntVar(value=0)

    # Image display function
    def update_image(idx=None):
        if idx is not None:
            current_idx.set(idx)
        ax.clear()
        im = ax.imshow(data[current_idx.get()],
                       cmap=cmap,
                       vmin=vmin.get(),
                       vmax=vmax.get())
        ax.set_title(f'{title or "Image"} {current_idx.get()} / {data.shape[0] - 1}')

        # Update colorbar
        cax.clear()
        fig.colorbar(im, cax=cax)
        canvas.draw()

    # Navigation functions
    def next_image():
        next_idx = (current_idx.get() + 1) % data.shape[0]
        update_image(next_idx)

    def prev_image():
        prev_idx = (current_idx.get() - 1) % data.shape[0]
        update_image(prev_idx)

    # Contrast adjustment functions
    def update_contrast(*args):
        update_image()

    def reset_contrast():
        vmin.set(data.min())
        vmax.set(data.max())
        update_image()

    # Create main container frames
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    controls_frame = ttk.Frame(main_frame)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Embed matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()

    # Add toolbar
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # Pack canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create contrast controls
    contrast_frame = ttk.LabelFrame(controls_frame, text="Contrast")
    contrast_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    ttk.Label(contrast_frame, text="Min:").pack(side=tk.LEFT, padx=5)
    min_entry = ttk.Entry(contrast_frame, textvariable=vmin, width=10)
    min_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(contrast_frame, text="Max:").pack(side=tk.LEFT, padx=5)
    max_entry = ttk.Entry(contrast_frame, textvariable=vmax, width=10)
    max_entry.pack(side=tk.LEFT, padx=5)

    ttk.Button(contrast_frame, text="Reset", command=reset_contrast).pack(side=tk.LEFT, padx=5)

    # Create navigation controls
    nav_frame = ttk.LabelFrame(controls_frame, text="Navigation")
    nav_frame.pack(side=tk.LEFT, padx=5)

    ttk.Button(nav_frame, text="Previous", command=prev_image).pack(side=tk.LEFT, padx=5)
    ttk.Button(nav_frame, text="Next", command=next_image).pack(side=tk.LEFT, padx=5)

    # Bind contrast updates
    vmin.trace_add("write", update_contrast)
    vmax.trace_add("write", update_contrast)

    # Show first image
    update_image(0)

    # Start the event loop
    root.mainloop()


# Use it:
data = np.load('/data/mct/22203/results/P6_ReverseOrder/averaged_flatfields.npy')
view_images_tk(data, cmap='gray', title='Flatfield')