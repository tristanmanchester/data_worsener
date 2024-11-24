import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def setup_visualization():
    """
    Set up the visualization window for processing progress.

    Returns:
    --------
    tuple
        Figure, axes, and image objects for visualization
    """
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Processing Progress')
    img1 = ax1.imshow(np.zeros((512, 512)), cmap='gray')
    img2 = ax2.imshow(np.zeros((512, 512)), cmap='gray')
    ax1.set_title('Original Slice')
    ax2.set_title('Processed Slice')
    plt.tight_layout()

    return fig, (ax1, ax2), img1, img2


def update_display(img, ax, data, title):
    """
    Update the display with new image data.

    Parameters:
    -----------
    img : matplotlib.image.AxesImage
        Image object to update
    ax : matplotlib.axes.Axes
        Axes object to update
    data : ndarray
        New image data
    title : str
        New title for the axes
    """
    img.set_data(data)
    img.set_clim(vmin=np.min(data), vmax=np.max(data))
    ax.set_title(title)
    plt.draw()
    plt.pause(0.001)


def save_test_outputs(output_folder, original_slice, intermediate_slice, final_slice):
    """
    Save test outputs showing original and final processed slices.

    Parameters:
    -----------
    output_folder : str
        Path to output folder
    original_slice : ndarray
        Original CT slice
    intermediate_slice : ndarray
        Not used in visualization but kept for compatibility
    final_slice : ndarray
        Final processed slice
    """
    # Save individual images
    cv2.imwrite(os.path.join(output_folder, 'test_original.png'),
                original_slice)
    cv2.imwrite(os.path.join(output_folder, 'test_processed.png'),
                final_slice)

    # Create comparison figure
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(original_slice, cmap='gray')
    plt.title(f'Original\n{original_slice.shape}')
    plt.axis('off')  # Hide axes for cleaner look

    plt.subplot(122)
    plt.imshow(final_slice, cmap='gray')
    plt.title(f'Final Result\n{final_slice.shape}')
    plt.axis('off')  # Hide axes for cleaner look

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'test_comparison.png'),
                bbox_inches='tight',
                dpi=300)  # Higher DPI for better quality
    plt.close()