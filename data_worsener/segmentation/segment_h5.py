import os
import h5py
import cupy as cp
import numpy as np
import time
from tqdm import tqdm
from skimage.filters import threshold_multiotsu, gaussian
from skimage.transform import resize
from cupyx.scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2


def resize_slice(image, output_size=(256, 256)):
    """
    Resize an image slice using high-quality bicubic interpolation.

    Parameters:
    -----------
    image : ndarray
        Input image
    output_size : tuple, optional
        Desired output size (default: (256, 256))

    Returns:
    --------
    ndarray
        Resized image
    """
    return resize(image,
                  output_size,
                  order=3,  # bicubic interpolation
                  mode='reflect',
                  anti_aliasing=True,
                  preserve_range=True)


def get_thresholds(middle_slice):
    """
    Calculate thresholds using multi-Otsu method.

    Parameters:
    -----------
    middle_slice : ndarray
        Middle slice of the volume for threshold calculation

    Returns:
    --------
    ndarray
        Array of threshold values
    """
    print("Calculating thresholds using multi-Otsu method...")
    middle_slice_np = cp.asnumpy(middle_slice)
    middle_slice_resized = resize_slice(middle_slice_np)
    print(f"Resized middle slice from {middle_slice_np.shape} to {middle_slice_resized.shape}")

    thresholds = threshold_multiotsu(middle_slice_resized, classes=4)
    print(f"Found threshold values: {thresholds}")
    return cp.asarray(thresholds)


def apply_thresholds(chunk, thresholds):
    """
    Apply thresholds to a chunk at full resolution.

    Parameters:
    -----------
    chunk : ndarray
        Input data chunk
    thresholds : ndarray
        Threshold values to apply

    Returns:
    --------
    ndarray
        Thresholded chunk
    """
    chunk = gaussian_filter(chunk, sigma=1)
    chunk = cp.digitize(chunk, bins=thresholds)
    chunk = cp.where(chunk == 3, 1, 0)
    return chunk


def get_memory_usage():
    """Get current GPU memory usage."""
    pool = cp.get_default_memory_pool()
    return f"GPU memory used: {pool.used_bytes() / 1024 ** 2:.2f} MB"


def process_h5_files(input_folder, output_folder, num_chunks=8):
    """
    Process all H5 files in the input folder to create segmentation labels.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing input H5 files
    output_folder : str
        Path to folder where segmented files will be saved
    num_chunks : int, optional
        Number of chunks to process the data in (default: 8)
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Get list of H5 files
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    total_files = len(h5_files)
    print(f"Found {total_files} HDF5 files to process")

    # Process each file
    for file_idx, filename in enumerate(h5_files, 1):
        start_time = time.time()
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        print(f"\n[{file_idx}/{total_files}] Processing file: {filename}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")

        with h5py.File(input_path, 'r') as f:
            # Process middle slice for thresholds
            print("Loading middle slice for threshold calculation...")
            middle_index = f['data'].shape[0] // 2
            middle_slice = cp.asarray(f['data'][middle_index, :, :])
            print(f"Original middle slice shape: {middle_slice.shape}")
            print(get_memory_usage())

            # Calculate thresholds and free memory
            thresholds = get_thresholds(middle_slice)
            del middle_slice
            cp.get_default_memory_pool().free_all_blocks()
            print("Freed GPU memory after threshold calculation")
            print(get_memory_usage())

            # Setup chunked processing
            slices = f['data'].shape[0]
            chunk_size = slices // num_chunks
            print(f"\nProcessing {slices} slices in {num_chunks} chunks")
            print(f"Chunk size: {chunk_size} slices")

            # Create output file
            with h5py.File(output_path, 'w') as new_f:
                new_dataset = new_f.create_dataset('data',
                                                   shape=f['data'].shape,
                                                   dtype=cp.uint16)

                # Process chunks with progress bar
                with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = (i + 1) * chunk_size if i != num_chunks - 1 else slices

                        print(f"\nChunk {i + 1}/{num_chunks}")
                        print(f"Processing slices {start} to {end}")

                        # Process chunk
                        chunk = cp.asarray(f['data'][start:end, :, :])
                        print(f"Loaded chunk shape: {chunk.shape}")
                        print(get_memory_usage())

                        chunk_labels = apply_thresholds(chunk, thresholds)
                        print("Applied thresholds")
                        print(get_memory_usage())

                        new_dataset[start:end, :, :] = cp.asnumpy(chunk_labels)
                        print("Saved processed chunk")

                        # Cleanup
                        del chunk, chunk_labels
                        cp.get_default_memory_pool().free_all_blocks()
                        print("Freed GPU memory")
                        print(get_memory_usage())

                        pbar.update(1)

        elapsed_time = time.time() - start_time
        print(f"\nCompleted processing {filename}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per slice: {elapsed_time / slices:.3f} seconds")
        print("-" * 80)

    print("\nAll files processed successfully!")


# Add this to src/segmentation/segment_h5.py after the other functions

def test_middle_slice(input_folder, output_folder):
    """
    Test the segmentation pipeline on the middle slice of the first H5 file.

    Parameters:
    -----------
    input_folder : str
        Path to folder containing input H5 files
    output_folder : str
        Path to folder where test results will be saved
    """
    # Find first H5 file
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    if not h5_files:
        print("No H5 files found in input folder")
        return

    test_file = h5_files[0]
    print(f'Testing segmentation pipeline on middle slice of: {test_file}')

    # Create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_path = os.path.join(input_folder, test_file)

    with h5py.File(input_path, 'r') as f:
        # Get middle slice
        middle_idx = f['data'].shape[0] // 2
        print(f'Extracting middle slice (index {middle_idx}) from total {f["data"].shape[0]} slices')
        slice_data = f['data'][middle_idx]
        original_slice = slice_data.copy()

        print("Processing slice...")
        try:
            # Convert to CuPy array
            slice_data = cp.asarray(slice_data)

            # Calculate thresholds
            thresholds = get_thresholds(slice_data)

            # Apply thresholds
            segmented_slice = apply_thresholds(slice_data, thresholds)

            # Convert back to numpy for saving
            segmented_slice = cp.asnumpy(segmented_slice)

            # Save outputs
            output_original = os.path.join(output_folder, 'test_original.png')
            output_segmented = os.path.join(output_folder, 'test_segmented.png')
            output_comparison = os.path.join(output_folder, 'test_comparison.png')

            print(f'Saving original slice to: {output_original}')
            cv2.imwrite(output_original, original_slice)

            print(f'Saving segmented slice to: {output_segmented}')
            cv2.imwrite(output_segmented, (segmented_slice * 65535).astype(np.uint16))

            # Create comparison visualization
            plt.figure(figsize=(10, 5))

            plt.subplot(121)
            plt.imshow(original_slice, cmap='gray')
            plt.title(f'Original\n{original_slice.shape}')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(segmented_slice, cmap='gray')
            plt.title(f'Segmented\n{segmented_slice.shape}')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(output_comparison, bbox_inches='tight', dpi=300)
            plt.close()

            print(f'Saved comparison image to: {output_comparison}')

        except Exception as e:
            print(f'Error during processing: {str(e)}')
            raise
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    print('Test segmentation completed!')
