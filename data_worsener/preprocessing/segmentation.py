import cv2
import numpy as np
from sklearn.cluster import KMeans


def reduce_pore_contrast(image, reduce_contrast=True):
    """
    Reduce the contrast of internal pores in a CT image and extract specimen information.

    Parameters:
    -----------
    image : ndarray
        Input CT image
    reduce_contrast : bool, optional
        Whether to reduce the contrast of pores (default: True)

    Returns:
    --------
    tuple
        - processed_image : ndarray
            Image with reduced pore contrast
        - mask : ndarray
            Binary mask of the specimen
        - largest_contour : ndarray
            Contour points of the specimen edge
        - max_radius : float
            Maximum radius from centroid to edge
        - min_radius : float
            Minimum radius from centroid to edge
        - cluster_means : ndarray
            Mean values of the specimen and background clusters
    """
    # Reshape non-zero pixels for K-means
    reshaped_image = image[image != 0].reshape(-1, 1)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=2, n_init=3)
    labels = kmeans.fit_predict(reshaped_image)

    # Find mean values of specimen and background
    cluster_means = np.sort(kmeans.cluster_centers_, axis=0)

    # Calculate threshold value as midpoint between clusters
    threshold_value = np.mean(cluster_means)

    # Apply threshold
    ret, thresh = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh.astype("uint8"),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Get largest contour (assumed to be specimen)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find centroid
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Calculate radii
    distances = np.sqrt((largest_contour[:, 0, 0] - cx) ** 2 +
                        (largest_contour[:, 0, 1] - cy) ** 2)
    max_radius = np.max(distances)
    min_radius = np.min(distances)

    # Create specimen mask
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], 0, 1, -1)

    # Process pores if requested
    if reduce_contrast:
        # Create pore mask
        pores = np.ma.where((mask == 1) & (image < threshold_value),
                            (threshold_value / 2).astype("uint16"),
                            0)

        # Blur pores
        pores = cv2.blur(pores, (5, 5))

        # Add blurred pores back to image
        image = (image + pores * 1.1).astype("uint16")

    return image, mask, largest_contour, max_radius, min_radius, cluster_means