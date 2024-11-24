import cv2
import numpy as np
from random import randrange, choice


def create_rings(image, number_of_rings, max_radius, mask):
    """
    Create ring artifacts in a CT image.

    Parameters:
    -----------
    image : ndarray
        Input image
    number_of_rings : int
        Number of rings to create
    max_radius : float
        Maximum radius for ring placement
    mask : ndarray
        Binary mask of the specimen

    Returns:
    --------
    ndarray
        Image with added ring artifacts
    """
    image = image.astype(np.float32)
    h, w = image.shape

    ring_radii = int(max_radius * 1)
    number_of_rings = int(number_of_rings * max_radius / (h * 0.3))

    # Calculate ring placement probabilities
    indices = np.arange(1, ring_radii + 1)
    decay_factor = 0.01
    ring_placement = np.exp(-decay_factor * indices)
    ring_placement = ring_placement / np.sum(ring_placement)

    # Create rings
    for n in range(number_of_rings):
        temp_ring_image = np.ones_like(image).astype(np.float32)

        # Select radius based on probability distribution
        radius = np.random.choice(a=ring_radii, replace=False, p=ring_placement)

        # Random intensity variation
        deviation = 0.4
        base_intensity = np.random.uniform(1 - deviation, 1 + deviation)

        # Random thickness between 1 and 2 pixels
        thickness = randrange(1, 2)

        # Draw the ring
        cv2.circle(temp_ring_image,
                   center=(int(w * 0.5), int(h * 0.5)),
                   radius=radius,
                   color=base_intensity,
                   thickness=thickness)

        # Blur the ring and apply mask
        temp_ring_image = cv2.blur(temp_ring_image, (3, 3))
        temp_ring_image[mask == 0] = 1

        # Multiply with original image
        image = cv2.multiply(image, temp_ring_image)

    image = np.clip(image, 0, 65535)
    return image.astype("uint16")


def create_large_ring(image, min_radius, max_radius, mask):
    """
    Create a large ring artifact on the outer edge of a CT image.

    Parameters:
    -----------
    image : ndarray
        Input image
    min_radius : float
        Minimum radius of the specimen
    max_radius : float
        Maximum radius of the specimen
    mask : ndarray
        Binary mask of the specimen

    Returns:
    --------
    ndarray
        Image with added large ring artifact
    """
    # Create ring image
    ring_image = np.ones_like(image).astype(np.float32)

    # Calculate radius and blur parameters
    radius_int = int(min_radius + (max_radius - min_radius) / 2)
    blur_int = int((max_radius - min_radius))

    # Separate specimen and background
    specimen = (image * mask).astype(np.float32)
    background = (image * (1 - mask)).astype(np.float32)

    # Create and blur the ring
    cv2.circle(ring_image,
               (image.shape[1] // 2, image.shape[0] // 2),
               radius_int, 3, -1)
    ring_image = cv2.blur(ring_image, (blur_int, blur_int))

    # Apply ring to background and combine with specimen
    background *= ring_image
    full_image = background + specimen

    return full_image.astype(np.uint16)