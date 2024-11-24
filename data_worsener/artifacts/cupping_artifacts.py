import cv2
import numpy as np


def add_edge_cupping_artefact(image, mask, edge_of_specimen, radius=100, intensity=15000, blur=51):
    """
    Add a cupping artifact to the edge of a specimen in a CT image.

    Parameters:
    -----------
    image : ndarray
        Input image
    mask : ndarray
        Binary mask of the specimen
    edge_of_specimen : ndarray
        Contour points of the specimen edge
    radius : int, optional
        Thickness of the edge enhancement (default: 100)
    intensity : int, optional
        Intensity of the edge enhancement (default: 15000)
    blur : int, optional
        Size of the blur kernel (default: 51)

    Returns:
    --------
    ndarray
        Image with added cupping artifact
    """
    # Create empty image same size as input
    cupping_image = np.zeros_like(image)

    # Draw line with specified thickness around contour
    cv2.polylines(cupping_image, [edge_of_specimen],
                  isClosed=True,
                  color=intensity,
                  thickness=radius)

    # Blur the line
    cupping_image = cv2.blur(cupping_image, (blur, blur))

    # Add pixels only within the specimen mask
    indices = np.where(mask == 1)
    image[indices] += cupping_image[indices]

    return image