import cupy as cp
from cupyx.scipy.ndimage import median_filter


def remove_stripe_based_sorting(sinogram, size, dim=1):
    """
    Remove stripe artifacts from a sinogram using sorting-based method.

    Parameters:
    -----------
    sinogram : ndarray
        Input sinogram
    size : int
        Size of the median filter
    dim : int, optional
        Dimension for median filtering (1 or 2) (default: 1)

    Returns:
    --------
    ndarray
        Sinogram with reduced stripe artifacts
    """
    # Use the original sinogram
    transposed_sinogram = sinogram
    num_rows, num_cols = transposed_sinogram.shape

    # Generate column indices matrix
    column_indices = cp.arange(num_cols)
    repeated_indices = cp.tile(column_indices, (num_rows, 1))

    # Combine indices with sinogram values
    combined_matrix = cp.asarray(cp.dstack((repeated_indices, transposed_sinogram)))

    # Sort by sinogram values
    sorted_matrix = cp.asarray([row[row[:, 1].argsort()]
                                for row in combined_matrix])

    # Apply median filter
    if dim == 2:
        sorted_matrix[:, :, 1] = median_filter(sorted_matrix[:, :, 1],
                                               (size, size))
    else:
        sorted_matrix[:, :, 1] = median_filter(sorted_matrix[:, :, 1],
                                               (size, 1))

    # Restore original order
    resorted_matrix = cp.asarray([row[row[:, 0].argsort()]
                                  for row in sorted_matrix])

    return cp.transpose(resorted_matrix[:, :, 1])